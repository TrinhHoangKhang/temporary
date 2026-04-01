# Dataset Download Stuck? Here's the Guide

## Why is it stuck?

The script is likely **downloading a large file (2-5GB)** from Amazon servers. This can take:
- **Fast network (100+ Mbps)**: 5-10 minutes
- **Normal network (20-50 Mbps)**: 20-40 minutes  
- **Slow network (<20 Mbps)**: 1+ hours

## What to do while waiting

### Option 1: Check if it's actually downloading (Recommended)
Open **another terminal** and check:

```bash
# Check cache directory size
du -sh ./cache/AmazonReviews2014/Books/raw/

# Watch real-time progress
watch du -sh ./cache/AmazonReviews2014/Books/raw/
```

If the file size is **growing**, it's working! Be patient.

---

### Option 2: Use Verbose Mode (See What's Happening)
If you want to see what's happening, use the verbose script:

```bash
python scripts/01_download_and_preprocess_verbose.py --category Books --cache_dir ./cache
```

This will show:
- More detailed logging
- Better error messages
- Progress indicators

---

### Option 3: Run in Background
If you want to detach from the terminal:

```bash
# Run in background, save output to file
nohup python scripts/01_download_and_preprocess.py --category Books --cache_dir ./cache > download.log 2>&1 &

# Check progress in real-time
tail -f download.log
```

---

### Option 4: Use Smaller Category (For Testing)
If you just want to test the setup without waiting long, try a smaller category:

```bash
# Much smaller dataset (~50MB instead of 2-5GB)
python scripts/01_download_and_preprocess.py --category Digital_Music --cache_dir ./cache
```

Available smaller categories:
- `Digital_Music` (smallest)
- `Apps_for_Android`
- `Beauty`

---

## If it really seems stuck (not growing)

### Check 1: Network connectivity
```bash
# Test internet connection
ping -c 5 8.8.8.8

# Test if you can reach Amazon servers
wget --spider https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
```

### Check 2: Disk space
```bash
# Check available space
df -h

# Make sure you have >10GB free
```

### Check 3: Kill and retry
```bash
# Kill the process
Ctrl+C

# Wait a moment, then retry
python scripts/01_download_and_preprocess.py --category Books --cache_dir ./cache
```

The script will check for partially downloaded files and resume if possible!

---

## Expected timeline

| Step | Time | What it's doing |
|------|------|-----------------|
| 1-2 min | Initialize | Setting up, checking cache |
| 2-40 min | **Download** | **Downloading reviews file (2-5GB)** |
| 1-5 min | Parse | Reading and parsing downloaded file |
| 5-10 min | Process | Creating ID mappings, sequences |
| 5-15 min | Metadata | Processing item metadata |
| **Total** | **20-70 min** | Depends on network speed |

---

## Once it's done

You'll see output like:
```
✓ Dataset created in 1234.5s
[Dataset] AmazonReviews2014
	Number of users: 603668
	Number of items: 330809
	Number of interactions: 8898263
	Average item sequence length: 14.7

✓ Processed data saved to: ./cache/AmazonReviews2014/Books/processed
  - all_item_seqs.json
  - id_mapping.json
  - metadata.sentence.json

✓ Dataset download and preprocessing complete!
```

After that, future runs will be **instant** (just loads from cache)!

---

## Next steps after download

```bash
# Verify files were created
ls -lh cache/AmazonReviews2014/Books/processed/

# Now you can use this in your training code:
python your_training_script.py
```

---

## Still stuck? Try this

```bash
# Test with minimal config (no metadata)
python scripts/01_download_and_preprocess.py \
    --category Digital_Music \
    --metadata_mode none \
    --cache_dir ./cache

# If this works, the network is fine
# Then try full Books download
```

**TL;DR**: It's probably just downloading. Check file size with `du -sh ./cache/` in another terminal. Be patient! ☕
