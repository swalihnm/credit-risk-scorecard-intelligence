# One-time Kaggle credential setup

The Kaggle API needs a personal token — I cannot generate this for you since it requires
logging into your Kaggle account.

1. Go to https://www.kaggle.com/settings (Account tab) while logged in.
2. Click **Create New Token** under the API section. This downloads `kaggle.json`.
3. Go to https://www.kaggle.com/c/home-credit-default-risk/rules and click **I Understand and
   Accept** (competition data requires accepting the rules before the API will let you download it).
4. Place the downloaded file at: `C:\Users\swali\.kaggle\kaggle.json`
   (create the `.kaggle` folder if it doesn't exist).

Once that file is in place, tell me and I'll run `ingest/kaggle_download.py` to pull all
7 tables into `sources/home_credit/`.
