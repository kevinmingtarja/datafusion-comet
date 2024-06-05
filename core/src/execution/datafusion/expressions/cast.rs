// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{
    any::Any, cmp::max, fmt::{Display, Formatter}, hash::{Hash, Hasher}, sync::Arc
};

use crate::errors::{CometError, CometResult};
use arrow::{
    compute::{cast_with_options, CastOptions}, datatypes::{ArrowNativeType, i256, DecimalType}, record_batch::RecordBatch, util::display::FormatOptions
};
use arrow_array::{
    types::{ArrowPrimitiveType, Decimal128Type, Decimal256Type, Int16Type, Int32Type, Int64Type, Int8Type}, Array, ArrayRef, ArrowNativeTypeOp, BooleanArray, GenericStringArray, OffsetSizeTrait, PrimitiveArray
};
use arrow_schema::{DataType, Schema};
use datafusion::logical_expr::ColumnarValue;
use datafusion_common::{internal_err, Result as DataFusionResult, ScalarValue};
use datafusion_physical_expr::PhysicalExpr;
use num::{traits::CheckedNeg, CheckedSub, Integer, Num};

use crate::execution::datafusion::expressions::utils::{
    array_with_timezone, down_cast_any_ref, spark_cast,
};

static TIMESTAMP_FORMAT: Option<&str> = Some("%Y-%m-%d %H:%M:%S%.f");
static CAST_OPTIONS: CastOptions = CastOptions {
    safe: true,
    format_options: FormatOptions::new()
        .with_timestamp_tz_format(TIMESTAMP_FORMAT)
        .with_timestamp_format(TIMESTAMP_FORMAT),
};

#[derive(Debug, Hash, PartialEq, Clone, Copy)]
pub enum EvalMode {
    Legacy,
    Ansi,
    Try,
}

#[derive(Debug, Hash)]
pub struct Cast {
    pub child: Arc<dyn PhysicalExpr>,
    pub data_type: DataType,
    pub eval_mode: EvalMode,

    /// When cast from/to timezone related types, we need timezone, which will be resolved with
    /// session local timezone by an analyzer in Spark.
    pub timezone: String,
}

macro_rules! cast_utf8_to_int {
    ($array:expr, $eval_mode:expr, $array_type:ty, $cast_method:ident) => {{
        let len = $array.len();
        let mut cast_array = PrimitiveArray::<$array_type>::builder(len);
        for i in 0..len {
            if $array.is_null(i) {
                cast_array.append_null()
            } else if let Some(cast_value) = $cast_method($array.value(i).trim(), $eval_mode)? {
                cast_array.append_value(cast_value);
            } else {
                cast_array.append_null()
            }
        }
        let result: CometResult<ArrayRef> = Ok(Arc::new(cast_array.finish()) as ArrayRef);
        result
    }};
}

macro_rules! cast_utf8_to_decimal {
    ($array:expr, $precision:expr, $scale:expr, $eval_mode:expr, $array_type:ty, $cast_method:ident) => {{
        let len = $array.len();
        let mut cast_array = PrimitiveArray::<$array_type>::builder(len).with_precision_and_scale($precision, $scale)?; // TODO: remove ?
        for i in 0..len {
            if $array.is_null(i) {
                cast_array.append_null()
            } else if let Some(cast_value) = $cast_method($array.value(i).trim(), $precision, $scale, $eval_mode)? {
                println!("append_value: {}", cast_value);
                cast_array.append_value(cast_value);
            } else {
                cast_array.append_null()
            }
        }
        let result: CometResult<ArrayRef> = Ok(Arc::new(cast_array.finish()) as ArrayRef);
        result?
    }};
}

impl Cast {
    pub fn new(
        child: Arc<dyn PhysicalExpr>,
        data_type: DataType,
        eval_mode: EvalMode,
        timezone: String,
    ) -> Self {
        Self {
            child,
            data_type,
            timezone,
            eval_mode,
        }
    }

    pub fn new_without_timezone(
        child: Arc<dyn PhysicalExpr>,
        data_type: DataType,
        eval_mode: EvalMode,
    ) -> Self {
        Self {
            child,
            data_type,
            timezone: "".to_string(),
            eval_mode,
        }
    }

    fn cast_array(&self, array: ArrayRef) -> DataFusionResult<ArrayRef> {
        let to_type = &self.data_type;
        let array = array_with_timezone(array, self.timezone.clone(), Some(to_type));
        let from_type = array.data_type();
        let cast_result = match (from_type, to_type) {
            (DataType::Utf8, DataType::Boolean) => {
                Self::spark_cast_utf8_to_boolean::<i32>(&array, self.eval_mode)?
            }
            (DataType::LargeUtf8, DataType::Boolean) => {
                Self::spark_cast_utf8_to_boolean::<i64>(&array, self.eval_mode)?
            }
            (
                DataType::Utf8,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
            ) => Self::cast_string_to_int::<i32>(to_type, &array, self.eval_mode)?,
            (
                DataType::LargeUtf8,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
            ) => Self::cast_string_to_int::<i64>(to_type, &array, self.eval_mode)?,
            (
                DataType::Dictionary(key_type, value_type),
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
            ) if key_type.as_ref() == &DataType::Int32
                && (value_type.as_ref() == &DataType::Utf8
                    || value_type.as_ref() == &DataType::LargeUtf8) =>
            {
                // TODO: we are unpacking a dictionary-encoded array and then performing
                // the cast. We could potentially improve performance here by casting the
                // dictionary values directly without unpacking the array first, although this
                // would add more complexity to the code
                match value_type.as_ref() {
                    DataType::Utf8 => {
                        let unpacked_array =
                            cast_with_options(&array, &DataType::Utf8, &CAST_OPTIONS)?;
                        Self::cast_string_to_int::<i32>(to_type, &unpacked_array, self.eval_mode)?
                    }
                    DataType::LargeUtf8 => {
                        let unpacked_array =
                            cast_with_options(&array, &DataType::LargeUtf8, &CAST_OPTIONS)?;
                        Self::cast_string_to_int::<i64>(to_type, &unpacked_array, self.eval_mode)?
                    }
                    dt => unreachable!(
                        "{}",
                        format!("invalid value type {dt} for dictionary-encoded string array")
                    ),
                }
            },
            (
                DataType::Utf8, 
                DataType::Decimal128(_, _) | DataType::Decimal256(_, _)
            ) => Self::cast_string_to_decimal::<i32>(to_type, &array, self.eval_mode)?,
            (
                DataType::LargeUtf8, 
                DataType::Decimal128(_, _) | DataType::Decimal256(_, _)
            ) => Self::cast_string_to_decimal::<i64>(to_type, &array, self.eval_mode)?,
            (
                DataType::Dictionary(key_type, value_type),
                DataType::Decimal128(_, _) | DataType::Decimal256(_, _),
            ) if key_type.as_ref() == &DataType::Int32
                && (value_type.as_ref() == &DataType::Utf8
                    || value_type.as_ref() == &DataType::LargeUtf8) =>
            {
                // TODO: we are unpacking a dictionary-encoded array and then performing
                // the cast. We could potentially improve performance here by casting the
                // dictionary values directly without unpacking the array first, although this
                // would add more complexity to the code
                match value_type.as_ref() {
                    DataType::Utf8 => {
                        let unpacked_array =
                            cast_with_options(&array, &DataType::Utf8, &CAST_OPTIONS)?;
                        Self::cast_string_to_decimal::<i32>(to_type, &unpacked_array, self.eval_mode)?
                    }
                    DataType::LargeUtf8 => {
                        let unpacked_array =
                            cast_with_options(&array, &DataType::LargeUtf8, &CAST_OPTIONS)?;
                        Self::cast_string_to_decimal::<i64>(to_type, &unpacked_array, self.eval_mode)?
                    }
                    dt => unreachable!(
                        "{}",
                        format!("invalid value type {dt} for dictionary-encoded string array")
                    ),
                }
            },
            _ => {
                // when we have no Spark-specific casting we delegate to DataFusion
                cast_with_options(&array, to_type, &CAST_OPTIONS)?
            }
        };
        Ok(spark_cast(cast_result, from_type, to_type))
    }

    fn cast_string_to_int<OffsetSize: OffsetSizeTrait>(
        to_type: &DataType,
        array: &ArrayRef,
        eval_mode: EvalMode,
    ) -> CometResult<ArrayRef> {
        let string_array = array
            .as_any()
            .downcast_ref::<GenericStringArray<OffsetSize>>()
            .expect("cast_string_to_int expected a string array");

        let cast_array: ArrayRef = match to_type {
            DataType::Int8 => {
                cast_utf8_to_int!(string_array, eval_mode, Int8Type, cast_string_to_i8)?
            }
            DataType::Int16 => {
                cast_utf8_to_int!(string_array, eval_mode, Int16Type, cast_string_to_i16)?
            }
            DataType::Int32 => {
                cast_utf8_to_int!(string_array, eval_mode, Int32Type, cast_string_to_i32)?
            }
            DataType::Int64 => {
                cast_utf8_to_int!(string_array, eval_mode, Int64Type, cast_string_to_i64)?
            }
            dt => unreachable!(
                "{}",
                format!("invalid integer type {dt} in cast from string")
            ),
        };
        Ok(cast_array)
    }

    fn cast_string_to_decimal<OffsetSize: OffsetSizeTrait>(
        to_type: &DataType,
        array: &ArrayRef,
        eval_mode: EvalMode,
    ) -> CometResult<ArrayRef> {
        let string_array = array
            .as_any()
            .downcast_ref::<GenericStringArray<OffsetSize>>()
            .expect("cast_string_to_decimal expected a string array");
        println!("here: {}", to_type);
        let cast_array: ArrayRef = match to_type {
            DataType::Decimal128(precision, scale) => {
                cast_utf8_to_decimal!(string_array, *precision, *scale, eval_mode, Decimal128Type, cast_string_to_decimal128)
            }
            DataType::Decimal256(precision, scale) => {
                cast_utf8_to_decimal!(string_array, *precision, *scale, eval_mode, Decimal256Type, cast_string_to_decimal256)
            }
            dt => unreachable!(
                "{}",
                format!("invalid decimal type {dt} in cast from string")
            ),
        };
        Ok(cast_array)
    }

    fn spark_cast_utf8_to_boolean<OffsetSize>(
        from: &dyn Array,
        eval_mode: EvalMode,
    ) -> CometResult<ArrayRef>
    where
        OffsetSize: OffsetSizeTrait,
    {
        let array = from
            .as_any()
            .downcast_ref::<GenericStringArray<OffsetSize>>()
            .unwrap();

        let output_array = array
            .iter()
            .map(|value| match value {
                Some(value) => match value.to_ascii_lowercase().trim() {
                    "t" | "true" | "y" | "yes" | "1" => Ok(Some(true)),
                    "f" | "false" | "n" | "no" | "0" => Ok(Some(false)),
                    _ if eval_mode == EvalMode::Ansi => Err(CometError::CastInvalidValue {
                        value: value.to_string(),
                        from_type: "STRING".to_string(),
                        to_type: "BOOLEAN".to_string(),
                    }),
                    _ => Ok(None),
                },
                _ => Ok(None),
            })
            .collect::<Result<BooleanArray, _>>()?;

        Ok(Arc::new(output_array))
    }
}

/// Equivalent to org.apache.spark.unsafe.types.UTF8String.toByte
fn cast_string_to_i8(str: &str, eval_mode: EvalMode) -> CometResult<Option<i8>> {
    Ok(cast_string_to_int_with_range_check(
        str,
        eval_mode,
        "TINYINT",
        i8::MIN as i32,
        i8::MAX as i32,
    )?
    .map(|v| v as i8))
}

/// Equivalent to org.apache.spark.unsafe.types.UTF8String.toShort
fn cast_string_to_i16(str: &str, eval_mode: EvalMode) -> CometResult<Option<i16>> {
    Ok(cast_string_to_int_with_range_check(
        str,
        eval_mode,
        "SMALLINT",
        i16::MIN as i32,
        i16::MAX as i32,
    )?
    .map(|v| v as i16))
}

/// Equivalent to org.apache.spark.unsafe.types.UTF8String.toInt(IntWrapper intWrapper)
fn cast_string_to_i32(str: &str, eval_mode: EvalMode) -> CometResult<Option<i32>> {
    do_cast_string_to_int::<i32>(str, eval_mode, "INT", i32::MIN)
}

/// Equivalent to org.apache.spark.unsafe.types.UTF8String.toLong(LongWrapper intWrapper)
fn cast_string_to_i64(str: &str, eval_mode: EvalMode) -> CometResult<Option<i64>> {
    do_cast_string_to_int::<i64>(str, eval_mode, "BIGINT", i64::MIN)
}

fn cast_string_to_int_with_range_check(
    str: &str,
    eval_mode: EvalMode,
    type_name: &str,
    min: i32,
    max: i32,
) -> CometResult<Option<i32>> {
    match do_cast_string_to_int(str, eval_mode, type_name, i32::MIN)? {
        None => Ok(None),
        Some(v) if v >= min && v <= max => Ok(Some(v)),
        _ if eval_mode == EvalMode::Ansi => Err(invalid_value(str, "STRING", type_name)),
        _ => Ok(None),
    }
}

#[derive(PartialEq)]
enum State {
    SkipLeadingWhiteSpace,
    SkipTrailingWhiteSpace,
    ParseSignAndDigits,
    ParseFractionalDigits,
}

/// Equivalent to
/// - org.apache.spark.unsafe.types.UTF8String.toInt(IntWrapper intWrapper, boolean allowDecimal)
/// - org.apache.spark.unsafe.types.UTF8String.toLong(LongWrapper longWrapper, boolean allowDecimal)
fn do_cast_string_to_int<
    T: Num + PartialOrd + Integer + CheckedSub + CheckedNeg + From<i32> + Copy,
>(
    str: &str,
    eval_mode: EvalMode,
    type_name: &str,
    min_value: T,
) -> CometResult<Option<T>> {
    let len = str.len();
    if str.is_empty() {
        return none_or_err(eval_mode, type_name, str);
    }

    let mut result: T = T::zero();
    let mut negative = false;
    let radix = T::from(10);
    let stop_value = min_value / radix;
    let mut state = State::SkipLeadingWhiteSpace;
    let mut parsed_sign = false;

    for (i, ch) in str.char_indices() {
        // skip leading whitespace
        if state == State::SkipLeadingWhiteSpace {
            if ch.is_whitespace() {
                // consume this char
                continue;
            }
            // change state and fall through to next section
            state = State::ParseSignAndDigits;
        }

        if state == State::ParseSignAndDigits {
            if !parsed_sign {
                negative = ch == '-';
                let positive = ch == '+';
                parsed_sign = true;
                if negative || positive {
                    if i + 1 == len {
                        // input string is just "+" or "-"
                        return none_or_err(eval_mode, type_name, str);
                    }
                    // consume this char
                    continue;
                }
            }

            if ch == '.' {
                if eval_mode == EvalMode::Legacy {
                    // truncate decimal in legacy mode
                    state = State::ParseFractionalDigits;
                    continue;
                } else {
                    return none_or_err(eval_mode, type_name, str);
                }
            }

            let digit = if ch.is_ascii_digit() {
                (ch as u32) - ('0' as u32)
            } else {
                return none_or_err(eval_mode, type_name, str);
            };

            // We are going to process the new digit and accumulate the result. However, before
            // doing this, if the result is already smaller than the
            // stopValue(Integer.MIN_VALUE / radix), then result * 10 will definitely be
            // smaller than minValue, and we can stop
            if result < stop_value {
                return none_or_err(eval_mode, type_name, str);
            }

            // Since the previous result is greater than or equal to stopValue(Integer.MIN_VALUE /
            // radix), we can just use `result > 0` to check overflow. If result
            // overflows, we should stop
            let v = result * radix;
            let digit = (digit as i32).into();
            match v.checked_sub(&digit) {
                Some(x) if x <= T::zero() => result = x,
                _ => {
                    return none_or_err(eval_mode, type_name, str);
                }
            }
        }

        if state == State::ParseFractionalDigits {
            // This is the case when we've encountered a decimal separator. The fractional
            // part will not change the number, but we will verify that the fractional part
            // is well-formed.
            if ch.is_whitespace() {
                // finished parsing fractional digits, now need to skip trailing whitespace
                state = State::SkipTrailingWhiteSpace;
                // consume this char
                continue;
            }
            if !ch.is_ascii_digit() {
                return none_or_err(eval_mode, type_name, str);
            }
        }

        // skip trailing whitespace
        if state == State::SkipTrailingWhiteSpace && !ch.is_whitespace() {
            return none_or_err(eval_mode, type_name, str);
        }
    }

    if !negative {
        if let Some(neg) = result.checked_neg() {
            if neg < T::zero() {
                return none_or_err(eval_mode, type_name, str);
            }
            result = neg;
        } else {
            return none_or_err(eval_mode, type_name, str);
        }
    }

    Ok(Some(result))
}

fn cast_string_to_decimal128(str: &str, precision: u8, scale: i8, eval_mode: EvalMode) -> CometResult<Option<i128>> {
    let cast = match str {
        "" | "." | "-" | "+" => None,
        _ => {
            match parse_decimal::<Decimal128Type>(str, precision, scale) {
                Some(v) => Some(v),
                None => {
                    // if eval_mode == EvalMode::Ansi {
                    //     Err(invalid_value(str, "STRING", "DECIMAL"))
                    // } else {
                    //     Ok(None)
                    // }
                    Some(i128::ZERO)
                }
            }
        }
    };
    Ok(cast)
}

fn cast_string_to_decimal256(str: &str, precision: u8, scale: i8, eval_mode: EvalMode) -> CometResult<Option<i256>> {
    let cast = match str {
        "" | "." | "-" | "+" => None,
        _ => {
            match parse_decimal::<Decimal256Type>(str, precision, scale) {
                Some(v) => Some(v),
                None => {
                    // if eval_mode == EvalMode::Ansi {
                    //     Err(invalid_value(str, "STRING", "DECIMAL"))
                    // } else {
                    //     Ok(None)
                    // }
                    Some(i256::ZERO)
                }
            }
        }
    };
    Ok(cast)
}

/// Copied from arrow-rs, modified to replicate Spark's behavior
/// Parse the string format decimal value to i128/i256 format and checking the precision and scale.
/// The result value can't be out of bounds.
pub fn parse_decimal<T: DecimalType>(
    s: &str,
    precision: u8,
    scale: i8,
) -> Option<T::Native> {
    let mut result = T::Native::usize_as(0);
    let mut fractionals = 0;
    let mut digits: u8 = 0;
    let mut exponent = 0;
    let base = T::Native::usize_as(10);

    let bs = s.as_bytes();
    let (bs, negative) = match bs.first() {
        Some(b'-') => (&bs[1..], true),
        Some(b'+') => (&bs[1..], false),
        _ => (bs, false),
    };

    if bs.is_empty() {
        return None;
    }

    let mut bs = bs.iter();
    // Overflow checks are not required if 10^(precision - 1) <= T::MAX holds.
    // Thus, if we validate the precision correctly, we can skip overflow checks.
    while let Some(b) = bs.next() {
        match b {
            b'0'..=b'9' => {
                if digits == 0 && *b == b'0' {
                    // Ignore leading zeros.
                    continue;
                }
                digits += 1;
                result = result.mul_wrapping(base);
                result = result.add_wrapping(T::Native::usize_as((b - b'0') as usize));
            }
            b'.' => {
                let mut has_exp = false;
                let mut negative = false;

                for b in bs.by_ref() {
                    if *b == b'e' || *b == b'E' {
                        has_exp = true;
                        continue;
                    }
                    if has_exp {
                        if *b == b'-' {
                            negative = true;
                            continue;
                        } else if *b == b'+' {
                            continue;
                        }

                        if !b.is_ascii_digit() {
                            return None;
                        }
                        exponent = exponent * 10 + (b - b'0') as i8;
                    }
                    if !b.is_ascii_digit() {
                        return None;
                    }
                    if fractionals == scale {
                        // We have processed all the digits that we need. All that
                        // is left is to validate that the rest of the string contains
                        // valid digits.
                        continue;
                    }
                    fractionals += 1;
                    digits += 1;
                    result = result.mul_wrapping(base);
                    result = result.add_wrapping(T::Native::usize_as((b - b'0') as usize));
                }

                if !negative {
                    fractionals = max(0, fractionals - exponent);
                } else {
                    fractionals += exponent;
                }
                digits += exponent as u8;

                // Fail on "."
                if digits == 0 {
                    return None;
                }
            }
            b'e' | b'E' => {
                let mut negative = false;
                for b in bs.by_ref() {
                    if *b == b'-' {
                        negative = true;
                        continue;
                    } else if *b == b'+' {
                        continue;
                    }

                    if !b.is_ascii_digit() {
                        return None;
                    }
                    exponent = exponent * 10 + (b - b'0') as i8;
                }
                // multiply result with exponent
                if !negative {
                    fractionals = max(0, fractionals - exponent);
                } else {
                    // initial: fractionals = 1, digits = 1
                    // end: 
                    // 1.5e-3 = 1.5 / 10^3 = 0.0015
                    fractionals += exponent;
                }
                digits += exponent as u8;
            }
            _ => {
                // return Err(ArrowError::ParseError(format!(
                //     "can't parse the string value {s} to decimal"
                // )));
                return None;
            }
        }
    }

    if fractionals < scale {
        let exp = scale - fractionals;
        if exp as u8 + digits > precision {
            // return Err(ArrowError::ParseError("parse decimal overflow".to_string()));
            return None;
        }
        let mul = base.pow_wrapping(exp as _);
        result = result.mul_wrapping(mul);
    } else if digits > precision {
        // return Err(ArrowError::ParseError("parse decimal overflow".to_string()));
        return None;
    }

    Some(if negative {
        result.neg_wrapping()
    } else {
        result
    })
}

/// Either return Ok(None) or Err(CometError::CastInvalidValue) depending on the evaluation mode
#[inline]
fn none_or_err<T>(eval_mode: EvalMode, type_name: &str, str: &str) -> CometResult<Option<T>> {
    match eval_mode {
        EvalMode::Ansi => Err(invalid_value(str, "STRING", type_name)),
        _ => Ok(None),
    }
}

#[inline]
fn invalid_value(value: &str, from_type: &str, to_type: &str) -> CometError {
    CometError::CastInvalidValue {
        value: value.to_string(),
        from_type: from_type.to_string(),
        to_type: to_type.to_string(),
    }
}

impl Display for Cast {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cast [data_type: {}, timezone: {}, child: {}, eval_mode: {:?}]",
            self.data_type, self.timezone, self.child, &self.eval_mode
        )
    }
}

impl PartialEq<dyn Any> for Cast {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.child.eq(&x.child)
                    && self.timezone.eq(&x.timezone)
                    && self.data_type.eq(&x.data_type)
                    && self.eval_mode.eq(&x.eval_mode)
            })
            .unwrap_or(false)
    }
}

impl PhysicalExpr for Cast {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _: &Schema) -> DataFusionResult<DataType> {
        Ok(self.data_type.clone())
    }

    fn nullable(&self, _: &Schema) -> DataFusionResult<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> DataFusionResult<ColumnarValue> {
        let arg = self.child.evaluate(batch)?;
        match arg {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(self.cast_array(array)?)),
            ColumnarValue::Scalar(scalar) => {
                // Note that normally CAST(scalar) should be fold in Spark JVM side. However, for
                // some cases e.g., scalar subquery, Spark will not fold it, so we need to handle it
                // here.
                let array = scalar.to_array()?;
                let scalar = ScalarValue::try_from_array(&self.cast_array(array)?, 0)?;
                Ok(ColumnarValue::Scalar(scalar))
            }
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.child.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> datafusion_common::Result<Arc<dyn PhysicalExpr>> {
        match children.len() {
            1 => Ok(Arc::new(Cast::new(
                children[0].clone(),
                self.data_type.clone(),
                self.eval_mode,
                self.timezone.clone(),
            ))),
            _ => internal_err!("Cast should have exactly one child"),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let mut s = state;
        self.child.hash(&mut s);
        self.data_type.hash(&mut s);
        self.timezone.hash(&mut s);
        self.eval_mode.hash(&mut s);
        self.hash(&mut s);
    }
}

#[cfg(test)]
mod test {
    use arrow::datatypes::Decimal128Type;
    use arrow_array::{Array, PrimitiveArray};
    use std::result::Result;

    use super::{cast_string_to_i8, cast_string_to_decimal128, EvalMode};

    #[test]
    fn test_cast_string_as_i8() {
        // basic
        assert_eq!(
            cast_string_to_i8("127", EvalMode::Legacy).unwrap(),
            Some(127_i8)
        );
        // assert_eq!(cast_string_to_i8("128", EvalMode::Legacy).unwrap(), None);
        // assert!(cast_string_to_i8("128", EvalMode::Ansi).is_err());
        // // decimals
        // assert_eq!(
        //     cast_string_to_i8("0.2", EvalMode::Legacy).unwrap(),
        //     Some(0_i8)
        // );
        // assert_eq!(
        //     cast_string_to_i8(".", EvalMode::Legacy).unwrap(),
        //     Some(0_i8)
        // );
        // // TRY should always return null for decimals
        // assert_eq!(cast_string_to_i8("0.2", EvalMode::Try).unwrap(), None);
        // assert_eq!(cast_string_to_i8(".", EvalMode::Try).unwrap(), None);
        // // ANSI mode should throw error on decimal
        // assert!(cast_string_to_i8("0.2", EvalMode::Ansi).is_err());
        // assert!(cast_string_to_i8(".", EvalMode::Ansi).is_err());
    }

    #[test]
    fn test_cast_string_as_decimal128() {
        let mut builder = PrimitiveArray::<Decimal128Type>::builder(5).with_precision_and_scale(5, 2).unwrap();
        builder.append_value(12700_i128);
        builder.append_value(0_i128);
        builder.append_null();
        builder.append_value(127_i128);
        builder.append_value(127123_i128);
        let array = builder.finish();
        println!("test: {}", array.is_null(1));
        println!("test: {}", array.is_null(2));
        
        
        // assert_eq!(
        //     cast_string_to_decimal128("127", 10, 0, EvalMode::Legacy).unwrap(),
        //     Some(127_i128)
        // );
        assert_eq!(
            cast_string_to_decimal128("127.123", 10, 3, EvalMode::Legacy).unwrap(),
            Some(127123_i128)
        );
        // assert_eq!(
        //     cast_string_to_decimal128("", 10, 0, EvalMode::Legacy).unwrap(),
        //     None
        // );
        // assert_eq!(
        //     cast_string_to_decimal128(".", 10, 0, EvalMode::Legacy).unwrap(),
        //     None
        // );
        // assert_eq!(
        //     cast_string_to_decimal128("-", 10, 0, EvalMode::Legacy).unwrap(),
        //     None
        // );
        // assert_eq!(
        //     cast_string_to_decimal128("4e3", 10, 0, EvalMode::Legacy).unwrap(),
        //     Some(4000_i128)
        // );
        assert_eq!(
            cast_string_to_decimal128("1.5", 10, 2, EvalMode::Legacy).unwrap(),
            Some(150_i128)
        );
        assert_eq!(
            cast_string_to_decimal128("1.5E2", 10, 2, EvalMode::Legacy).unwrap(),
            Some(150_i128)
        );
        assert_eq!(
            cast_string_to_decimal128("127123e-3", 10, 3, EvalMode::Legacy).unwrap(),
            Some(127123_i128)
        );
    }
}
