#!/usr/bin/env python3
"""
Migration Script: Supabase Storage → Google Cloud Storage
This script helps migrate your codebase from Supabase Storage to GCS.

Usage:
    python migrate_to_gcs.py --dry-run  # Show what would change (no modifications)
    python migrate_to_gcs.py            # Apply changes
    python migrate_to_gcs.py --file api.py  # Migrate specific file only
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# File patterns to search
PYTHON_FILES = ['**/*.py']
TYPESCRIPT_FILES = ['website/src/**/*.ts', 'website/src/**/*.tsx']

# Replacement patterns
PYTHON_REPLACEMENTS = [
    # Import statement
    (
        r'from supabase import create_client',
        r'from supabase import create_client\nfrom storage_client import storage_client'
    ),
    # Download operations
    (
        r'supabase\.storage\.from_\(["\']([^"\']+)["\']\)\.download\(([^)]+)\)',
        r'storage_client.download(\1, \2)'
    ),
    # Upload operations (file path)
    (
        r'supabase\.storage\.from_\(["\']([^"\']+)["\']\)\.upload\(([^,]+),\s*([^)]+)\)',
        r'storage_client.upload_from_file(\1, \2, \3)'
    ),
    # Upload operations with options (bytes/data)
    (
        r'supabase\.storage\.from_\(["\']([^"\']+)["\']\)\.upload\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
        r'storage_client.upload_from_bytes(\1, \2, \3, content_type=None)  # TODO: Extract content_type from \4'
    ),
    # Remove/delete operations
    (
        r'supabase\.storage\.from_\(["\']([^"\']+)["\']\)\.remove\(([^)]+)\)',
        r'storage_client.delete(\1, \2)'
    ),
    # Get public URL
    (
        r'supabase\.storage\.from_\(["\']([^"\']+)["\']\)\.getPublicUrl\(([^)]+)\)',
        r'storage_client.get_public_url(\1, \2)'
    ),
]

TYPESCRIPT_REPLACEMENTS = [
    # Import statement
    (
        r"import.*from '@/integrations/supabase/client'",
        r"import { supabase } from '@/integrations/supabase/client';\nimport { storageClient } from '@/utils/storageClient';"
    ),
    # Download operations
    (
        r"supabase\.storage\.from\(['\"]([^'\"]+)['\"]\)\.download\(([^)]+)\)",
        r"storageClient.download('\1', \2)"
    ),
    # Upload operations
    (
        r"supabase\.storage\.from\(['\"]([^'\"]+)['\"]\)\.upload\(([^,]+),\s*([^,]+)(?:,\s*([^)]+))?\)",
        r"storageClient.upload('\1', \2, \3, \4 ?? undefined)"
    ),
    # Remove operations
    (
        r"supabase\.storage\.from\(['\"]([^'\"]+)['\"]\)\.remove\(([^)]+)\)",
        r"storageClient.delete('\1', \2)"
    ),
    # Get public URL
    (
        r"supabase\.storage\.from\(['\"]([^'\"]+)['\"]\)\.getPublicUrl\(([^)]+)\)",
        r"storageClient.getPublicUrl('\1', \2)"
    ),
    # Create signed URL
    (
        r"supabase\.storage\.from\(['\"]([^'\"]+)['\"]\)\.createSignedUrl\(([^,]+),\s*([^)]+)\)",
        r"storageClient.createSignedUrl('\1', \2, \3)"
    ),
]

def find_files(pattern: str, root: str = '.') -> List[Path]:
    """Find files matching pattern."""
    files = []
    for path in Path(root).glob(pattern):
        if path.is_file():
            files.append(path)
    return files

def replace_in_file(file_path: Path, replacements: List[Tuple[str, str]], dry_run: bool = False) -> Tuple[int, List[str]]:
    """Apply replacements to a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return 0, [f"  ⚠️  Skipped {file_path} (binary file)"]
    
    original_content = content
    changes = []
    
    for pattern, replacement in replacements:
        matches = re.finditer(pattern, content)
        for match in matches:
            line_num = original_content[:match.start()].count('\n') + 1
            old_text = match.group(0)
            new_text = re.sub(pattern, replacement, old_text)
            
            if old_text != new_text:
                changes.append(f"  Line {line_num}: {old_text[:50]}...")
        
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return len([c for c in changes if c]), changes
    return 0, []

def migrate_file(file_path: Path, file_type: str, dry_run: bool = False) -> dict:
    """Migrate a single file."""
    if file_type == 'python':
        replacements = PYTHON_REPLACEMENTS
    elif file_type == 'typescript':
        replacements = TYPESCRIPT_REPLACEMENTS
    else:
        return {'file': str(file_path), 'changed': False, 'changes': []}
    
    num_changes, change_details = replace_in_file(file_path, replacements, dry_run)
    
    return {
        'file': str(file_path),
        'changed': num_changes > 0,
        'num_changes': num_changes,
        'changes': change_details
    }

def main():
    parser = argparse.ArgumentParser(description='Migrate Supabase Storage to Google Cloud Storage')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying files')
    parser.add_argument('--file', type=str, help='Migrate specific file only')
    parser.add_argument('--python-only', action='store_true', help='Only migrate Python files')
    parser.add_argument('--typescript-only', action='store_true', help='Only migrate TypeScript files')
    args = parser.parse_args()
    
    mode = "DRY RUN" if args.dry_run else "MIGRATION"
    print(f"\n{'='*60}")
    print(f"  {mode}: Supabase Storage → Google Cloud Storage")
    print(f"{'='*60}\n")
    
    results = []
    
    # Find files to migrate
    files_to_migrate = []
    
    if args.file:
        # Single file
        file_path = Path(args.file)
        if file_path.exists():
            if file_path.suffix == '.py':
                files_to_migrate.append((file_path, 'python'))
            elif file_path.suffix in ['.ts', '.tsx']:
                files_to_migrate.append((file_path, 'typescript'))
        else:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
    else:
        # Find all files
        if not args.typescript_only:
            for pattern in PYTHON_FILES:
                files_to_migrate.extend([(f, 'python') for f in find_files(pattern)])
        
        if not args.python_only:
            for pattern in TYPESCRIPT_FILES:
                files_to_migrate.extend([(f, 'typescript') for f in find_files(pattern)])
    
    if not files_to_migrate:
        print("⚠️  No files found to migrate.")
        return
    
    print(f"Found {len(files_to_migrate)} file(s) to check...\n")
    
    # Migrate each file
    changed_count = 0
    for file_path, file_type in files_to_migrate:
        result = migrate_file(file_path, file_type, args.dry_run)
        results.append(result)
        
        if result['changed']:
            changed_count += 1
            status = "✅ MODIFIED" if not args.dry_run else "🔍 WOULD CHANGE"
            print(f"{status}: {result['file']} ({result['num_changes']} change(s))")
            for change in result['changes'][:3]:  # Show first 3 changes
                print(change)
            if len(result['changes']) > 3:
                print(f"  ... and {len(result['changes']) - 3} more")
            print()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Files checked: {len(files_to_migrate)}")
    print(f"  Files {'would be changed' if args.dry_run else 'changed'}: {changed_count}")
    
    if args.dry_run:
        print(f"\n  💡 Run without --dry-run to apply changes")
    else:
        print(f"\n  ✅ Migration complete!")
        print(f"  ⚠️  Please review changes and test thoroughly")
        print(f"  📝 Manual fixes may be needed for:")
        print(f"     - Complex upload operations with options")
        print(f"     - Error handling differences")
        print(f"     - Authentication/signed URLs")
    
    print()

if __name__ == '__main__':
    main()

