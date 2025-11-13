#!/bin/bash
#
# Repository Cleanup Script
# Cleans up temporary files, organizes outputs, and maintains a clean structure
#

set -e

echo "=================================================================================="
echo "                    REPOSITORY CLEANUP SCRIPT"
echo "=================================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
REMOVED_COUNT=0
MOVED_COUNT=0
KEPT_COUNT=0

echo "Step 1: Cleaning log files..."
if ls *.log 1> /dev/null 2>&1; then
    echo -e "${YELLOW}Found log files:${NC}"
    ls -lh *.log
    echo ""
    read -p "Remove these log files? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f *.log
        echo -e "${GREEN}✓ Log files removed${NC}"
        REMOVED_COUNT=$((REMOVED_COUNT + $(ls *.log 2>/dev/null | wc -l)))
    fi
else
    echo "No log files found"
fi
echo ""

echo "Step 2: Cleaning old experiment outputs..."
if [ -f "experiment_output.txt" ]; then
    SIZE=$(du -h experiment_output.txt | cut -f1)
    echo -e "${YELLOW}Found experiment_output.txt ($SIZE)${NC}"
    read -p "Remove this file? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f experiment_output.txt
        echo -e "${GREEN}✓ Removed experiment_output.txt${NC}"
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
    fi
else
    echo "No old experiment output found"
fi
echo ""

echo "Step 3: Cleaning Python cache..."
if find . -type d -name "__pycache__" 2>/dev/null | grep -q .; then
    CACHE_COUNT=$(find . -type d -name "__pycache__" | wc -l)
    echo -e "${YELLOW}Found $CACHE_COUNT __pycache__ directories${NC}"
    read -p "Remove Python cache directories? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        find . -type f -name "*.pyo" -delete 2>/dev/null || true
        echo -e "${GREEN}✓ Python cache cleaned${NC}"
        REMOVED_COUNT=$((REMOVED_COUNT + CACHE_COUNT))
    fi
else
    echo "No Python cache found"
fi
echo ""

echo "Step 4: Cleaning pytest cache..."
if [ -d ".pytest_cache" ]; then
    echo -e "${YELLOW}Found .pytest_cache directory${NC}"
    read -p "Remove pytest cache? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .pytest_cache
        echo -e "${GREEN}✓ Pytest cache removed${NC}"
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
    fi
else
    echo "No pytest cache found"
fi
echo ""

echo "Step 5: Organizing test outputs..."
if [ -d "test_outputs" ]; then
    TEST_FILES=$(find test_outputs -type f | wc -l)
    echo -e "${YELLOW}Found test_outputs directory with $TEST_FILES files${NC}"
    read -p "Keep test outputs? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        rm -rf test_outputs
        echo -e "${GREEN}✓ Test outputs removed${NC}"
        REMOVED_COUNT=$((REMOVED_COUNT + TEST_FILES))
    else
        echo -e "${GREEN}✓ Keeping test outputs${NC}"
        KEPT_COUNT=$((KEPT_COUNT + TEST_FILES))
    fi
else
    echo "No test outputs directory found"
fi
echo ""

echo "Step 6: Checking old result directories..."
if [ -d "results" ]; then
    RESULT_DIRS=$(find results -maxdepth 1 -type d | tail -n +2 | wc -l)
    if [ $RESULT_DIRS -gt 3 ]; then
        echo -e "${YELLOW}Found $RESULT_DIRS result directories${NC}"
        echo "Keeping most recent 3, removing older ones..."
        read -p "Proceed? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Keep newest 3, remove older
            find results -maxdepth 1 -type d | tail -n +2 | sort -r | tail -n +4 | xargs rm -rf 2>/dev/null || true
            echo -e "${GREEN}✓ Old results cleaned (kept 3 most recent)${NC}"
            REMOVED_COUNT=$((REMOVED_COUNT + RESULT_DIRS - 3))
        fi
    else
        echo "Result directories: $RESULT_DIRS (keeping all)"
        KEPT_COUNT=$((KEPT_COUNT + RESULT_DIRS))
    fi
fi
echo ""

echo "Step 7: Checking for duplicate/backup files..."
BACKUP_FILES=$(find . -maxdepth 2 -type f \( -name "*.bak" -o -name "*~" -o -name "*.swp" \) 2>/dev/null | wc -l)
if [ $BACKUP_FILES -gt 0 ]; then
    echo -e "${YELLOW}Found $BACKUP_FILES backup files${NC}"
    find . -maxdepth 2 -type f \( -name "*.bak" -o -name "*~" -o -name "*.swp" \)
    read -p "Remove these files? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find . -maxdepth 2 -type f \( -name "*.bak" -o -name "*~" -o -name "*.swp" \) -delete
        echo -e "${GREEN}✓ Backup files removed${NC}"
        REMOVED_COUNT=$((REMOVED_COUNT + BACKUP_FILES))
    fi
else
    echo "No backup files found"
fi
echo ""

echo "Step 8: Organizing documentation..."
DOC_FILES=$(ls -1 *.md 2>/dev/null | wc -l)
if [ $DOC_FILES -gt 5 ]; then
    echo -e "${YELLOW}Found $DOC_FILES markdown files in root${NC}"
    ls -1 *.md
    echo ""
    echo "Consider moving some to a docs/ directory"
    read -p "Create docs/ directory and organize? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p docs
        # Keep README.md and QUICKSTART.md in root, move others
        for file in SUMMARY.md TEST_CLEANUP_SUMMARY.md VISUALIZATIONS.md; do
            if [ -f "$file" ]; then
                mv "$file" docs/
                echo "  Moved $file → docs/"
                MOVED_COUNT=$((MOVED_COUNT + 1))
            fi
        done
        echo -e "${GREEN}✓ Documentation organized${NC}"
    fi
else
    echo "Documentation files: $DOC_FILES (no action needed)"
fi
echo ""

echo "=================================================================================="
echo "                         CLEANUP SUMMARY"
echo "=================================================================================="
echo -e "${GREEN}Files removed:  $REMOVED_COUNT${NC}"
echo -e "${YELLOW}Files moved:    $MOVED_COUNT${NC}"
echo -e "Files kept:     $KEPT_COUNT"
echo ""

echo "Final repository structure:"
tree -L 2 -I '.venv|.git|__pycache__|*.pyc' --dirsfirst || ls -1

echo ""
echo "=================================================================================="
echo -e "${GREEN}Cleanup complete!${NC}"
echo "=================================================================================="
