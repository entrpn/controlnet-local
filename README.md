## setup

```bash
pip install -r requirements.txt
git clone https://github.com/lllyasviel/ControlNet.git
```

Then you can generate images:

```bash
python main.py --image-uri anime_girl.png --model canny --prompt "anime girl" --steps 50 --kwargs '{"low_threshold" : 150, "high_threshold":200}'
```