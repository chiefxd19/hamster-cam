# 🐹 Hamster Mirror - Traced Edition

The cursed MS-Paint hamster that mirrors your face & hands. Exactly like the TikTok filter video.

Left = your webcam with MediaPipe hand landmarks  
Right = hamster sprite that reacts

![demo](https://github.com/user-attachments/assets/demo-placeholder)

### Traced Sprites
All 9 sprites were traced directly from the original video into clean transparent PNGs:
- `neutral` - default blob
- `smile` - blush happy
- `laugh` - big mouth open / tongue out
- `thumbs_down` - thumbs down
- `thumbs_up` - fist up
- `peace` - pointing / peace sign
- `tutu` - ballerina with star wand
- `nerd` - glasses
- `pleading` - hands clasped

### Install & Run
```bash
git clone https://github.com/YOUR_USERNAME/hamster-mirror.git
cd hamster-mirror
pip install -r requirements.txt
python main.py
```
Press `ESC` to quit.

### How it works
- `MediaPipe FaceMesh` -> smile / mouth open detection -> maps to hamster face
- `MediaPipe Hands` -> gesture classifier (fist, thumbs_down, peace, etc) -> maps to hamster sprite
- Sprites are overlaid with alpha on white background, with `#ID (name) NK@conf` label like original

### Upload to GitHub
1. Create new repo on github.com (don't init with README)
2. Drag & drop all files from this folder
3. Commit

No build needed - pure Python.