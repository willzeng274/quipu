// TODO: Errors and types

use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::PathBuf;

/// Cross-platform file path representation that can be safely serialized to/from binary.
///
/// This struct handles the complexity of file path encoding across different platforms:
/// - **Unix**: Paths are UTF-8, stored directly as bytes
/// - **Windows**: Paths may be UTF-16, stored with encoding prefix (0xFE = UTF-8, 0xFF = UTF-16)
///
/// Uses Vec<u8> for serialization and deserialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FilePath {
    pub bytes: Vec<u8>,
}

impl FilePath {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn encoding_info(&self) -> &'static str {
        if self.bytes.is_empty() {
            return "empty path";
        }

        #[cfg(windows)]
        match self.bytes[0] {
            Self::UTF8_MARKER => "Windows UTF-8",
            Self::UTF16_MARKER => "Windows UTF-16",
            _ => "Windows legacy/unknown",
        }

        #[cfg(unix)]
        {
            "Unix UTF-8"
        }
    }

    #[cfg(windows)]
    pub const UTF8_MARKER: u8 = 0xFE;
    #[cfg(windows)]
    pub const UTF16_MARKER: u8 = 0xFF;

    #[cfg(unix)]
    pub fn encode_os_string(os_string: OsString) -> Vec<u8> {
        // unix = utf-8
        os_string.into_encoded_bytes()
    }

    #[cfg(unix)]
    pub fn decode_to_os_string(bytes: &[u8]) -> OsString {
        // unix = utf-8
        unsafe { OsString::from_encoded_bytes_unchecked(bytes.to_vec()) }
    }

    #[cfg(windows)]
    pub fn encode_os_string(os_string: OsString) -> Vec<u8> {
        // windows = utf-8 or utf-16
        let (utf8_bytes, utf16_bytes) = os_string.into_encoded_bytes();

        if let Some(utf16_bytes) = utf16_bytes {
            // utf-16
            let mut result = vec![Self::UTF16_MARKER];
            result.extend(utf16_bytes);
            result
        } else {
            // utf-8
            let mut result = vec![Self::UTF8_MARKER];
            result.extend(utf8_bytes);
            result
        }
    }

    #[cfg(windows)]
    pub fn decode_to_os_string(bytes: &[u8]) -> OsString {
        if bytes.is_empty() {
            return OsString::new();
        }

        match bytes[0] {
            Self::UTF8_MARKER => {
                // utf-8
                let utf8_bytes = &bytes[1..];
                unsafe { OsString::from_encoded_bytes_unchecked(utf8_bytes.to_vec()) }
            }
            Self::UTF16_MARKER => {
                // utf-16
                let utf16_bytes = &bytes[1..];

                if utf16_bytes.len() % 2 != 0 {
                    // corrupted data, fallback to utf-8
                    return unsafe { OsString::from_encoded_bytes_unchecked(bytes[1..].to_vec()) };
                }

                let utf16_units: Vec<u16> = utf16_bytes
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                unsafe { OsString::from_encoded_bytes_unchecked(utf16_units) }
            }
            _ => {
                // legacy format or unknown marker, fallback to utf-8
                unsafe { OsString::from_encoded_bytes_unchecked(bytes.to_vec()) }
            }
        }
    }
}

impl From<PathBuf> for FilePath {
    fn from(path: PathBuf) -> Self {
        let os_string = path.into_os_string();
        let bytes = Self::encode_os_string(os_string);
        Self { bytes }
    }
}

impl From<FilePath> for PathBuf {
    fn from(file_path: FilePath) -> Self {
        let os_string = FilePath::decode_to_os_string(&file_path.bytes);
        PathBuf::from(os_string)
    }
}

impl From<OsString> for FilePath {
    fn from(os_string: OsString) -> Self {
        let bytes = Self::encode_os_string(os_string);
        Self { bytes }
    }
}

impl From<FilePath> for OsString {
    fn from(file_path: FilePath) -> Self {
        FilePath::decode_to_os_string(&file_path.bytes)
    }
}