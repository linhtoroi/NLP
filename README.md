*** My Implementation ***

I implemented the main model, and code training model

src/model - model.py
        - Embedding.py
        - Encoder.py
        - Decoder.py

NL2SQL.py

*** How run code ***

virtualenv -p /usr/bin/python3.7 env 

source venv/bin/activate

pip install torch torchvision

python3 -m pip install -r requirements.txt

export PYTHONPATH=`pwd` && python -m nltk.downloader punkt


python3 run.py
- Init model:
model = NL2SQLmodel(args)
- Use cuda:
model.model.cuda()
- Train:
model.train()
