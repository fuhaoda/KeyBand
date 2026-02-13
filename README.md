# KeyBand Studio

KeyBand Studio is a ready-to-run numbered keyboard web app with **realistic piano + guitar + flute**.
It lets anyone play **1–7** on a normal keyboard.

All audio files are already inside this repo. No download, no rebuild.

---

## Super Simple Start (No Experience Needed)

### Option A: Double-click (Mac)

Double-click the file named **start_local.command** in the KeyBand folder.

Then open this in your browser:

```
http://127.0.0.1:8080
```

Click **Start Audio** and play.

### Option B: Terminal (still simple)

1. Open Terminal inside the KeyBand folder
2. Run:

```bash
python3 -m http.server 8080
```

3. Open:

```
http://127.0.0.1:8080
```

---

## How to Play

- Notes: `1–7` (top row) or numpad `1–7`
- Octave up: hold `Shift`
- Octave down: hold `Ctrl`
- Ped: hold `8` or `9` or `0` or `I` or `O` or `Space`
- Switch instrument: use the dropdown (Piano / Guitar / Flute)
- Mobile Keyboard: tap the button, hold Oct+ or Oct- to shift temporarily, press both to exit

---

## If You See “Directory listing”

Stop the server with `Ctrl + C`, then start it again **inside the KeyBand folder**:

```bash
python3 -m http.server 8080
```

---

## Home Server (Basement)

Run this once:

```bash
docker compose up -d --build
```

Then open in a browser:

```
http://<your-server-ip>:8080
```

 
