//! Minimal reader for MATLAB Level-5 MAT-files (the format written by `save`
//! with the default `-v7` / `-v6` options).
//!
//! Supported: numeric arrays of every class, logical, char, cell and struct
//! arrays, compressed (`miCOMPRESSED`) elements, both byte orders.
//! Not supported: sparse/object arrays (skipped with an error) and the
//! HDF5-based `-v7.3` format (detected and reported).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use flate2::read::ZlibDecoder;

use crate::error::{Error, Result};

// Data types (MAT-file "mi" types).
const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;
const MI_UTF32: u32 = 18;

// Array classes ("mx" classes).
const MX_CELL: u8 = 1;
const MX_STRUCT: u8 = 2;
const MX_OBJECT: u8 = 3;
const MX_CHAR: u8 = 4;
const MX_SPARSE: u8 = 5;

/// A decoded MATLAB value.
#[derive(Debug, Clone)]
pub enum MatValue {
    /// Numeric or logical array, stored column-major as f64 (real part only).
    Numeric { dims: Vec<usize>, data: Vec<f64> },
    /// Character array decoded to a string (rows joined with '\n').
    Char { dims: Vec<usize>, text: String },
    /// Cell array, column-major.
    Cell { dims: Vec<usize>, items: Vec<MatValue> },
    /// Struct array: `elements[i]` maps field name -> value, column-major.
    Struct { dims: Vec<usize>, fields: Vec<String>, elements: Vec<BTreeMap<String, MatValue>> },
    /// A class this reader does not interpret (sparse, object, function handle…).
    Unsupported { class: u8 },
}

impl MatValue {
    pub fn dims(&self) -> &[usize] {
        match self {
            MatValue::Numeric { dims, .. }
            | MatValue::Char { dims, .. }
            | MatValue::Cell { dims, .. }
            | MatValue::Struct { dims, .. } => dims,
            MatValue::Unsupported { .. } => &[],
        }
    }

    /// Scalar numeric value, if this is a non-empty numeric array.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            MatValue::Numeric { data, .. } => data.first().copied(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MatValue::Char { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }

    /// Numeric 2-D matrix as (rows, cols, column-major data).
    pub fn as_matrix(&self) -> Option<(usize, usize, &[f64])> {
        match self {
            MatValue::Numeric { dims, data } => {
                let rows = dims.first().copied().unwrap_or(0);
                let cols = if dims.len() > 1 { dims[1..].iter().product() } else { 1 };
                Some((rows, cols, data.as_slice()))
            }
            _ => None,
        }
    }
}

/// All top-level variables of a MAT-file, in file order.
#[derive(Debug, Clone)]
pub struct MatFile {
    pub header: String,
    pub variables: Vec<(String, MatValue)>,
}

impl MatFile {
    pub fn get(&self, name: &str) -> Option<&MatValue> {
        self.variables.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

pub fn read_mat_file(path: &Path) -> Result<MatFile> {
    let bytes = std::fs::read(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
    parse_mat(&bytes).map_err(|e| e.with_context(&path.display().to_string()))
}

pub fn parse_mat(bytes: &[u8]) -> Result<MatFile> {
    if bytes.len() < 128 {
        return Err(Error::Format("file shorter than the 128-byte MAT header".into()));
    }
    let header = String::from_utf8_lossy(&bytes[..116]).trim_end().to_string();
    if header.starts_with("MATLAB 7.3") || &bytes[..8] == b"\x89HDF\r\n\x1a\n" {
        return Err(Error::Format(
            "this is a MATLAB v7.3 (HDF5) file; please re-save it in MATLAB with `save(..., '-v7')`".into(),
        ));
    }
    if !header.starts_with("MATLAB 5.0 MAT-file") {
        return Err(Error::Format("not a MATLAB Level-5 MAT-file".into()));
    }
    let endian = match &bytes[126..128] {
        b"IM" => Endian::Little,
        b"MI" => Endian::Big,
        _ => return Err(Error::Format("unknown byte-order marker in MAT header".into())),
    };

    let mut variables = Vec::new();
    let mut cur = Cursor { buf: &bytes[128..], pos: 0, endian };
    while cur.remaining() >= 8 {
        let (ty, size, small) = cur.tag()?;
        if small.is_some() {
            continue; // stray small element at top level: ignore
        }
        let payload = cur.take(size)?;
        match ty {
            MI_COMPRESSED => {
                let mut inflated = Vec::new();
                ZlibDecoder::new(payload)
                    .read_to_end(&mut inflated)
                    .map_err(|e| Error::Format(format!("zlib decompression failed: {e}")))?;
                let mut inner = Cursor { buf: &inflated, pos: 0, endian };
                while inner.remaining() >= 8 {
                    let (ity, isize, ismall) = inner.tag()?;
                    if ismall.is_some() {
                        continue;
                    }
                    let ip = inner.take(isize)?;
                    inner.align8();
                    if ity == MI_MATRIX {
                        variables.push(parse_matrix(ip, endian)?);
                    }
                }
            }
            MI_MATRIX => variables.push(parse_matrix(payload, endian)?),
            _ => {}
        }
        cur.align8();
    }
    Ok(MatFile { header, variables })
}

#[derive(Clone, Copy, Debug)]
enum Endian {
    Little,
    Big,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    endian: Endian,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn u32_at(&self, at: usize) -> u32 {
        let b = [self.buf[at], self.buf[at + 1], self.buf[at + 2], self.buf[at + 3]];
        match self.endian {
            Endian::Little => u32::from_le_bytes(b),
            Endian::Big => u32::from_be_bytes(b),
        }
    }

    /// Reads a data-element tag. Returns (type, byte size, small-element payload).
    fn tag(&mut self) -> Result<(u32, usize, Option<&'a [u8]>)> {
        if self.remaining() < 8 {
            return Err(Error::Format("truncated data element tag".into()));
        }
        let first = self.u32_at(self.pos);
        // Small data element: upper 16 bits hold the byte count (1..=4).
        let small_size = match self.endian {
            Endian::Little => first >> 16,
            Endian::Big => first >> 16,
        };
        if small_size != 0 {
            let ty = first & 0xFFFF;
            let data = &self.buf[self.pos + 4..self.pos + 4 + (small_size as usize).min(4)];
            self.pos += 8;
            return Ok((ty, small_size as usize, Some(data)));
        }
        let size = self.u32_at(self.pos + 4) as usize;
        self.pos += 8;
        Ok((first, size, None))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::Format(format!(
                "data element claims {n} bytes but only {} remain",
                self.remaining()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn align8(&mut self) {
        let rem = self.pos % 8;
        if rem != 0 {
            self.pos = (self.pos + 8 - rem).min(self.buf.len());
        }
    }

    /// Reads one sub-element (tag + payload, honouring small-element packing).
    fn element(&mut self) -> Result<(u32, &'a [u8])> {
        let (ty, size, small) = self.tag()?;
        if let Some(d) = small {
            return Ok((ty, &d[..size.min(d.len())]));
        }
        let data = self.take(size)?;
        self.align8();
        Ok((ty, data))
    }
}

fn decode_numbers(ty: u32, data: &[u8], endian: Endian) -> Result<Vec<f64>> {
    macro_rules! conv {
        ($t:ty, $n:expr) => {{
            data.chunks_exact($n)
                .map(|c| {
                    let mut a = [0u8; $n];
                    a.copy_from_slice(c);
                    (match endian {
                        Endian::Little => <$t>::from_le_bytes(a),
                        Endian::Big => <$t>::from_be_bytes(a),
                    }) as f64
                })
                .collect()
        }};
    }
    Ok(match ty {
        MI_INT8 => data.iter().map(|&b| b as i8 as f64).collect(),
        MI_UINT8 | MI_UTF8 => data.iter().map(|&b| b as f64).collect(),
        MI_INT16 => conv!(i16, 2),
        MI_UINT16 | MI_UTF16 => conv!(u16, 2),
        MI_INT32 => conv!(i32, 4),
        MI_UINT32 | MI_UTF32 => conv!(u32, 4),
        MI_SINGLE => conv!(f32, 4),
        MI_DOUBLE => conv!(f64, 8),
        MI_INT64 => conv!(i64, 8),
        MI_UINT64 => conv!(u64, 8),
        other => return Err(Error::Format(format!("unexpected numeric data type {other}"))),
    })
}

fn parse_matrix(payload: &[u8], endian: Endian) -> Result<(String, MatValue)> {
    if payload.is_empty() {
        // Empty miMATRIX (used for [] inside cells/structs).
        return Ok((String::new(), MatValue::Numeric { dims: vec![0, 0], data: vec![] }));
    }
    let mut cur = Cursor { buf: payload, pos: 0, endian };

    let (_, flags) = cur.element()?;
    if flags.len() < 8 {
        return Err(Error::Format("array flags sub-element too short".into()));
    }
    let flag_word = Cursor { buf: flags, pos: 0, endian }.u32_at(0);
    let class = (flag_word & 0xFF) as u8;
    let is_complex = flag_word & 0x0800 != 0;

    let (dty, dims_raw) = cur.element()?;
    let dims: Vec<usize> = decode_numbers(dty, dims_raw, endian)?.into_iter().map(|d| d.max(0.0) as usize).collect();
    let count: usize = dims.iter().product();

    let (_, name_raw) = cur.element()?;
    let name = String::from_utf8_lossy(name_raw).trim_end_matches('\0').to_string();

    let value = match class {
        MX_CELL => {
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                let (ty, data) = cur.element()?;
                if ty != MI_MATRIX {
                    return Err(Error::Format("cell element is not miMATRIX".into()));
                }
                items.push(parse_matrix(data, endian)?.1);
            }
            MatValue::Cell { dims, items }
        }
        MX_STRUCT | MX_OBJECT => {
            if class == MX_OBJECT {
                let _class_name = cur.element()?;
            }
            let (lty, lraw) = cur.element()?;
            let name_len = decode_numbers(lty, lraw, endian)?.first().copied().unwrap_or(0.0) as usize;
            let (_, names_raw) = cur.element()?;
            let fields: Vec<String> = if name_len == 0 {
                vec![]
            } else {
                names_raw
                    .chunks(name_len)
                    .map(|c| String::from_utf8_lossy(c).trim_end_matches('\0').to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            };
            let mut elements = Vec::with_capacity(count);
            for _ in 0..count {
                let mut el = BTreeMap::new();
                for f in &fields {
                    let (ty, data) = cur.element()?;
                    if ty != MI_MATRIX {
                        return Err(Error::Format(format!("struct field '{f}' is not miMATRIX")));
                    }
                    el.insert(f.clone(), parse_matrix(data, endian)?.1);
                }
                elements.push(el);
            }
            MatValue::Struct { dims, fields, elements }
        }
        MX_CHAR => {
            let (ty, data) = if cur.remaining() > 0 { cur.element()? } else { (MI_UINT16, &[][..]) };
            let codes: Vec<u32> = match ty {
                MI_UTF8 => {
                    let s = String::from_utf8_lossy(data).to_string();
                    s.chars().map(|c| c as u32).collect()
                }
                _ => decode_numbers(ty, data, endian)?.into_iter().map(|v| v as u32).collect(),
            };
            let rows = dims.first().copied().unwrap_or(0).max(1);
            let cols = if count == 0 { 0 } else { count / rows };
            // Column-major -> row strings.
            let mut lines = Vec::with_capacity(rows);
            for r in 0..rows {
                let line: String = (0..cols)
                    .filter_map(|c| codes.get(c * rows + r).and_then(|&u| char::from_u32(u)))
                    .collect();
                lines.push(line);
            }
            MatValue::Char { dims, text: lines.join("\n") }
        }
        MX_SPARSE => MatValue::Unsupported { class },
        6..=15 => {
            let (ty, data) = if cur.remaining() > 0 { cur.element()? } else { (MI_DOUBLE, &[][..]) };
            let mut values = decode_numbers(ty, data, endian)?;
            if is_complex {
                // Keep the real part only; imaginary data follows and is ignored.
            }
            values.truncate(count.max(values.len().min(count)));
            MatValue::Numeric { dims, data: values }
        }
        _ => MatValue::Unsupported { class },
    };
    Ok((name, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_input() {
        assert!(parse_mat(&[0u8; 10]).is_err());
    }
}
