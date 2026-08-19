# sakuhart

**Turn a video into mosaic art.** Every tile is a different photograph out of your own folder.

<img src="https://raw.githubusercontent.com/hinanohart/sakuhart/main/docs/still_zoom.jpg" width="100%" alt="A mosaic with one cell zoomed in to the photograph that fills it">

```bash
pip install sakuhart
sakuhart --demo                    # rebuild the bundled painting, no arguments
sakuhart clip.mp4 -t ./dogs        # your video, your photos
```

Python 3.10+. Video also needs `ffmpeg`; still images do not.

The full README, the moving demo, and ready-made photo packs are on GitHub:
**<https://github.com/hinanohart/sakuhart>**

MIT.
