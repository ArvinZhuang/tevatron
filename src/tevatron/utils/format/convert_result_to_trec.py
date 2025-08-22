from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--remove_query', action='store_true')
args = parser.parse_args()

with open(args.input) as f_in, open(args.output, 'w') as f_out:
    cur_qid = None
    rank = 0
    for line in f_in:
        try:
            qid, docid, score = line.split()
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        if cur_qid != qid:
            cur_qid = qid
            rank = 0
        if args.remove_query and qid == docid:
            continue
        rank += 1      
        f_out.write(f'{qid} Q0 {docid} {rank} {score} dense\n')
