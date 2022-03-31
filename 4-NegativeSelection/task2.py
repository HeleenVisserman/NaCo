from sklearn.metrics import roc_auc_score
import numpy as np
import subprocess

lines = []
filenames = []
n = 10

def processFile(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f]

        chunks = []
        lineSize = []
        count = 0
        for line in lines:
            chunks += [[line[i: i+n] for i in range(0,len(line)-n+1,n)]]
            lineSize += [len(chunks[count])]
            count += 1
        print(len(chunks))
        print(lineSize)
    
        # Create chunk.test file with chunks on every line
        test = filename.split("/")
        write_filename = "syscalls/generated/chunks_" + test[len(test)-1]
        with open(write_filename, 'w') as w:
            for l in chunks:
                for c in l:
                    w.write("%s\n" % c)
        # create an array that indicates how many lines in the chunk.test belong to which label. --> this way we can also later recombine the scores of the chunks that belong to the same original line.
        linesize_filename = "syscalls/generated/ls_" + test[len(test)-1]
        with open(linesize_filename, 'w') as ls:
            for size in lineSize:
                ls.write("%s\n" % size)
    

def processTrainData(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f]
        chunks = []
        for line in lines:
            chunks += [[line[i: i+n] for i in range(0,len(line)-n+1,n)]]
        print(len(chunks))

        test = filename.split("/")

        write_filename = "chunks_" + test[len(test)-1]
        with open(write_filename, 'w') as w:
            for item in chunks:
                w.write("%s\n" % item)
    
# ========== file processing commands cert =====================
# processFile("./syscalls/snd-cert/snd-cert.1.test")
# processFile("./syscalls/snd-cert/snd-cert.2.test")
# processFile("./syscalls/snd-cert/snd-cert.3.test")
# processTrainData("./syscalls/snd-cert/snd-cert.train")

# ========== file processing commands unm =====================
# processFile("./syscalls/snd-unm/snd-unm.1.test")
# processFile("./syscalls/snd-unm/snd-unm.2.test")
# processFile("./syscalls/snd-unm/snd-unm.3.test")
# processTrainData("./syscalls/snd-unm/snd-unm.train")

def processResults(resultfile, linesfile, labelsfile):
    with open(resultfile) as f:
        with open(linesfile) as ls:
            lines = [line.strip() for line in f]
            lineSizes = [line.strip() for line in ls]
            resultValues = []

            index = 0
            for size in lineSizes:
                value = 0.0
                sz = int(size)
                for i in range(sz):
                    value += float(lines[i+index])
                resultValues += [value/sz] if sz != 0 else [0.0]
                index += sz
            print(resultValues)
            print(len(resultValues))

            with open(labelsfile) as lab:
                labels = [int(line.strip()) for line in lab]
                score = roc_auc_score(labels, resultValues)
                print("ROCAUC: ", score)


# =========== Result processing commands cert ========================
# processResults("./syscalls/results/cert/cert2-result-r3.txt", "./syscalls/generated/ls_snd-cert.2.test", "./syscalls/snd-cert/snd-cert.2.labels")   
# processResults("./syscalls/results/cert/cert1-result-r4.txt", "./syscalls/generated/ls_snd-cert.1.test", "./syscalls/snd-cert/snd-cert.1.labels")   
# processResults("./syscalls/results/cert/cert3-result-r4.txt", "./syscalls/generated/ls_snd-cert.3.test", "./syscalls/snd-cert/snd-cert.3.labels")   
# processResults("./syscalls/results/cert/cert1-result-r3.txt", "./syscalls/generated/ls_snd-cert.1.test", "./syscalls/snd-cert/snd-cert.1.labels")   
# processResults("./syscalls/results/cert/cert1-result-r5.txt", "./syscalls/generated/ls_snd-cert.1.test", "./syscalls/snd-cert/snd-cert.1.labels")   

# =========== Result processing commands unm ========================
# processResults("./syscalls/results/unm/unm1-result-r4.txt", "./syscalls/generated/ls_snd-unm.1.test", "./syscalls/snd-unm/snd-unm.1.labels")
# processResults("./syscalls/results/unm/unm2-result-r4.txt", "./syscalls/generated/ls_snd-unm.2.test", "./syscalls/snd-unm/snd-unm.2.labels")   
# processResults("./syscalls/results/unm/unm3-result-r4.txt", "./syscalls/generated/ls_snd-unm.3.test", "./syscalls/snd-unm/snd-unm.3.labels")     




