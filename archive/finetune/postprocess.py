import penman
from tqdm import tqdm
from model_interface.tokenization_bart import AMRBartTokenizer

tokenizer = AMRBartTokenizer.from_pretrained("../models/mbart-en-id-smaller-pre-trained-fine-tune")

root_path = f"/home/nafkhanzam/kode/nafkhanzam/thesis/old_evals/tnp"
file_path = f"{root_path}/generated_predictions.txt"
out_path = f"{root_path}/output.amr"

with open(file_path, "r") as f:
    lines = [line.replace(" </AMR>", "").replace("Ġ", "") for line in f]

amrs = []

for line in tqdm(lines):
    line_splitted = line.split()
    res = tokenizer.tokenize_amr(line_splitted)
    graph, _, _ = tokenizer.decode_amr(res)
    amr = penman.encode(graph)
    amrs.append(amr)

with open(out_path, "w") as p_writer:
    p_writer.write("\n\n".join(amrs))
