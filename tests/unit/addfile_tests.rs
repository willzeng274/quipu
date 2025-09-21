use quipu::{AddFile, FileType, create_mock_embedding};
use std::path::PathBuf;

#[test]
fn test_addfile_new() {
    let filename_embedding = create_mock_embedding("test.txt", "filename");
    let addfile = AddFile::new(PathBuf::from("/path/to/test.txt"), FileType::File, filename_embedding);

    assert_eq!(PathBuf::from(addfile.file_path), PathBuf::from("/path/to/test.txt"));
    assert_eq!(addfile.file_type, FileType::File);
    assert_eq!(addfile.mime_type, None);
    assert_eq!(addfile.permissions, None);
    assert_eq!(addfile.file_size, 0);
    assert!(addfile.created_time.is_some());
    assert!(addfile.accessed_time.is_some());
    assert!(addfile.modified_time > 0);
}

#[test]
fn test_addfile_builder_pattern() {
    let filename_embedding = create_mock_embedding("document.pdf", "filename");
    let content_embedding = create_mock_embedding("content", "content");

    let addfile = AddFile::new(PathBuf::from("/docs/document.pdf"), FileType::File, filename_embedding)
        .mime_type("application/pdf")
        .permissions(0o644)
        .file_size(1024)
        .tags(&["important", "work"])
        .content_embedding(content_embedding)
        .content_hash("content_hash");

    assert_eq!(PathBuf::from(addfile.file_path), PathBuf::from("/docs/document.pdf"));
    assert_eq!(addfile.mime_type, Some("application/pdf".to_string()));
    assert_eq!(addfile.permissions, Some(0o644));
    assert_eq!(addfile.file_size, 1024);
    assert_eq!(addfile.tags, Some(vec!["important".to_string(), "work".to_string()]));
    assert!(addfile.content_embedding.is_some());
    assert_eq!(addfile.content_hash, Some("content_hash".to_string()));
}

#[test]
fn test_addfile_chaining() {
    let filename_embedding = create_mock_embedding("file.txt", "filename");

    let addfile = AddFile::new(PathBuf::from("/path/file.txt"), FileType::File, filename_embedding)
        .mime_type("text/plain")
        .permissions(0o755)
        .file_size(2048)
        .modified_time(1234567890)
        .accessed_time(1234567891)
        .created_time(1234567889)
        .tags(&["test"]);

    assert_eq!(addfile.mime_type, Some("text/plain".to_string()));
    assert_eq!(addfile.permissions, Some(0o755));
    assert_eq!(addfile.file_size, 2048);
    assert_eq!(addfile.modified_time, 1234567890);
    assert_eq!(addfile.accessed_time, Some(1234567891));
    assert_eq!(addfile.created_time, Some(1234567889));
    assert_eq!(addfile.tags, Some(vec!["test".to_string()]));
}

#[test]
fn test_addfile_directory() {
    let filename_embedding = create_mock_embedding("mydir", "filename");

    let addfile = AddFile::new(PathBuf::from("/path/to/mydir"), FileType::Directory, filename_embedding)
        .permissions(0o755)
        .tags(&["directory", "important"]);

    assert_eq!(addfile.file_type, FileType::Directory);
    assert_eq!(addfile.permissions, Some(0o755));
    assert_eq!(addfile.tags, Some(vec!["directory".to_string(), "important".to_string()]));
}

#[test]
fn test_addfile_symlink() {
    let filename_embedding = create_mock_embedding("link", "filename");

    let addfile = AddFile::new(PathBuf::from("/path/to/link"), FileType::Symlink, filename_embedding)
        .permissions(0o777);

    assert_eq!(addfile.file_type, FileType::Symlink);
    assert_eq!(addfile.permissions, Some(0o777));
}
