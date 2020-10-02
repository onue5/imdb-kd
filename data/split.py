import csv

# read
with open("./IMDB Dataset.csv") as fp:
    csv.read = csv.reader(fp)
    next(csv.read)

    lines = [line for line in csv.read]

# train
with open("./train.csv", "w") as fp:
    csv.writerow = csv.writer(fp)
    csv.writerow.writerow(['review', 'sentiment'])
    for line in lines[:48000]:
        csv.writerow.writerow(line)

# test
with open("./test.csv", "w") as fp:
    csv.writerow = csv.writer(fp)
    csv.writerow.writerow(['review', 'sentiment'])
    for line in lines[48000:]:
        csv.writerow.writerow(line)
