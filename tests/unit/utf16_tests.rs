#[cfg(windows)]
mod windows_utf16_tests {
    use quipu::FilePath;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn test_utf16_simple_roundtrip() {
        let utf16_chars: Vec<u16> = vec![
            0x0043, 0x003A, 0x005C, // C:\
            0x0074, 0x0065, 0x0073, 0x0074, // test
            0x002E, 0x0074, 0x0078, 0x0074  // .txt
        ];

        let os_string = unsafe { OsString::from_encoded_bytes_unchecked(utf16_chars) };
        let original = PathBuf::from(os_string);

        let file_path = FilePath::from(original.clone());
        let converted_back = PathBuf::from(file_path);

        assert_eq!(original, converted_back);
        assert_eq!(file_path.bytes[0], FilePath::UTF16_MARKER);
    }

    #[test]
    fn test_utf16_encoding_detection() {
        let utf16_chars: Vec<u16> = vec![0x0043, 0x003A, 0x005C, 0x0074, 0x0065, 0x0073, 0x0074]; // C:\test
        let os_string = unsafe { OsString::from_encoded_bytes_unchecked(utf16_chars) };
        let path = PathBuf::from(os_string);

        let file_path = FilePath::from(path);
        assert_eq!(file_path.bytes[0], FilePath::UTF16_MARKER);
    }

    #[test]
    fn test_utf8_vs_utf16_detection() {
        let utf8_path = PathBuf::from("C:\\simple.txt");
        let utf8_file_path = FilePath::from(utf8_path);
        assert_eq!(utf8_file_path.bytes[0], FilePath::UTF8_MARKER);

        let utf16_chars: Vec<u16> = vec![0x0043, 0x003A, 0x005C, 0xD83D, 0xDE80]; // C:\🚀
        let utf16_os = unsafe { OsString::from_encoded_bytes_unchecked(utf16_chars) };
        let utf16_path = PathBuf::from(utf16_os);

        let utf16_file_path = FilePath::from(utf16_path);
        assert_eq!(utf16_file_path.bytes[0], FilePath::UTF16_MARKER);
    }

    #[test]
    fn test_utf16_corrupted_data_fallback() {
        let corrupted_bytes = vec![FilePath::UTF16_MARKER, 0x41, 0x42, 0x43];
        let file_path = FilePath::from_bytes(corrupted_bytes);

        let _ = PathBuf::from(file_path);
    }

    #[test]
    fn test_utf16_empty_after_marker() {
        let empty_utf16 = vec![FilePath::UTF16_MARKER];
        let file_path = FilePath::from_bytes(empty_utf16);

        let converted_back = PathBuf::from(file_path);
        assert_eq!(converted_back, PathBuf::new());
    }

    #[test]
    fn test_utf16_invalid_marker_fallback() {
        let invalid_marker = vec![0xAA, 0x41, 0x42, 0x43];
        let file_path = FilePath::from_bytes(invalid_marker);

        let converted_back = PathBuf::from(file_path);
        assert!(converted_back.to_string_lossy().contains("ABC"));
    }

    #[test]
    fn test_utf16_os_string_direct_conversion() {
        let utf16_chars: Vec<u16> = vec![0x0048, 0x0065, 0x006C, 0x006C, 0x006F];
        let original_os = unsafe { OsString::from_encoded_bytes_unchecked(utf16_chars) };

        let file_path = FilePath::from(original_os.clone());
        let converted_back = OsString::from(file_path);

        assert_eq!(original_os, converted_back);
    }

    #[test]
    fn test_utf16_roundtrip_consistency() {
        let utf16_chars: Vec<u16> = vec![
            0x0043, 0x003A, 0x005C, 0x0066, 0x006F, 0x006C, 0x0064, 0x0065, 0x0072 // C:\folder
        ];

        let original_os = unsafe { OsString::from_encoded_bytes_unchecked(utf16_chars) };
        let original_path = PathBuf::from(original_os.clone());

        let file_path1 = FilePath::from(original_path.clone());
        let path1_back = PathBuf::from(file_path1.clone());
        let file_path2 = FilePath::from(path1_back.clone());
        let path2_back = PathBuf::from(file_path2.clone());

        assert_eq!(original_path, path1_back);
        assert_eq!(original_path, path2_back);
        assert_eq!(file_path1.bytes, file_path2.bytes);
    }
}