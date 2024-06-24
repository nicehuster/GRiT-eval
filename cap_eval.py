
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


result_json_file = "results.json"
pred_results = json.load(open(result_json_file))
res, gts = {}, {}
for preds in pred_results:
    gt_captions = preds["gt_captions"]
    pred_captions = preds["pred_captions"]
    img_id = preds["image_id"]
    for i, (gt, pred) in enumerate(zip(gt_captions, pred_captions)):
        key = f"{img_id}_region_{i}"
        res[key] = [{"caption": pred[0]}]
        gts[key] = [{"caption": gt}]


tokenizer = PTBTokenizer()
res, gts = tokenizer.tokenize(res), tokenizer.tokenize(gts)
# Evaluate results for each metric.
for metric in (Cider(), Meteor(), Bleu(),Rouge(),Spice()):
    kwargs = {"verbose": 0} if isinstance(metric, Bleu) else {}
    score, _ = metric.compute_score(gts, res, **kwargs)
    print(metric.method(), score)
