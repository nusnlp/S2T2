#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, argparse, glob, math

def calc_distribution(annotations) :
    count_O = 0
    count_T = 0
    count_X = 0

    for annotation in annotations:
        if annotation['breakdown'] == 'O':
            count_O += 1
        elif annotation['breakdown'] == 'T':
            count_T += 1
        elif annotation['breakdown'] == 'X':
            count_X += 1

    prob_O = count_O*1.0 / (count_O + count_T + count_X)
    prob_T = count_T*1.0 / (count_O + count_T + count_X)
    prob_X = count_X*1.0 / (count_O + count_T + count_X)

    return [prob_O, prob_T, prob_X]

def majority_label(prob_O, prob_T, prob_X, threshold):

    if prob_O >= prob_T and prob_O >= prob_X and prob_O >= threshold:
        return "O"
    elif prob_T >= prob_O and prob_T >= prob_X and prob_T >= threshold:
        return "T"
    elif prob_X >= prob_T and prob_X >= prob_O and prob_X >= threshold:
        return "X"
    else:
        return "O"

def majority_label_lenient(prob_O, prob_T, prob_X, threshold):

    if prob_O >= prob_T + prob_X and prob_O >= threshold:
        return "O_l"
    elif prob_T + prob_X >= prob_O and prob_T + prob_X >= threshold:
        return "X_l"
    else:
        return "O_l"

def kld(p, q):
    k = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            k += p[i] * (math.log(p[i]/q[i],2))

    return k

def jsd(p, q):
    m = []
    for i in range(len(p)):
        m.append((p[i]+q[i])/2.0)

    return (kld(p,m) + kld(q,m)) / 2.0

def mse(p, q):
    total = 0.0

    for i in range(len(p)):
        total += pow(p[i] - q[i],2)

    return total / len(p)

def main():

    file_num = 0
    label_num = 0

    correct_num = 0
    incorrect_num = 0

    predO_ansO = 0
    predO_ansT = 0
    predO_ansX = 0
    predT_ansO = 0
    predT_ansT = 0
    predT_ansX = 0
    predX_ansO = 0
    predX_ansT = 0
    predX_ansX = 0

    predO_ansO_l = 0
    predO_ansX_l = 0
    predT_ansO_l = 0
    predT_ansX_l = 0
    predX_ansO_l = 0
    predX_ansX_l = 0

    jsd_O_T_X_sum = 0.0
    jsd_O_TX_sum = 0.0
    jsd_OT_X_sum = 0.0
    mse_O_T_X_sum = 0.0
    mse_O_TX_sum = 0.0
    mse_OT_X_sum = 0.0

    pred_dist_list = []
    ans_dist_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='datapath', action='store', metavar='TESTDIR', required=True,
                        help='(XXX contains XXX/*.log.json) テスト用の対話データが入ったディレクトリ')
    parser.add_argument('-o', dest='evalpath', action='store', metavar='EVALDIR', required=True,
                        help='(XXX contains XXX/*.labels.json) 評価用データが入ったディレクトリ')
    parser.add_argument('-t', dest='threshold', action='store', metavar='THRESHOLD', required=False,
                        help='(threshold for labels) 正解ラベルを決定するためのしきい値')

    args = parser.parse_args()

    data_path = args.datapath
    eval_path = args.evalpath

    threshold = 0.0
    if args.threshold != None:
        threshold = float(args.threshold)

    if not data_path.endswith('/') or not data_path.endswith('\\'):
        data_path += '/'
    if not eval_path.endswith('/') or not eval_path.endswith('\\'):
        eval_path += '/'

    data_files = glob.glob(data_path + '*log.json')

    for f in data_files:
        file_num += 1

        data_fp = open(f, "r")
        data_json = json.load(data_fp)
        data_fp.close()

        dlg_id = data_json["dialogue-id"]
        eval_fp = open(eval_path + dlg_id + ".labels.json", "r")
        eval_json = json.load(eval_fp)
        eval_fp.close()

        eval_index = 0
        for turn in data_json['turns']:
            if turn['speaker'] == "U" or turn['annotations']==[]:
                continue

            label_num += 1

            ans_prob_dist = calc_distribution(turn['annotations'])
            ans_dist_list.append(ans_prob_dist)
            ans_label = majority_label(ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold)

            ans_label_l = majority_label_lenient(ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold)

            target_label = eval_json['turns'][eval_index]['labels'][0]

            pred_prob_dist = [float(target_label['prob-O']), float(target_label['prob-T']), float(target_label['prob-X'])]
            pred_dist_list.append(pred_prob_dist)

            jsd_O_T_X_sum += jsd(ans_prob_dist,pred_prob_dist)
            jsd_O_TX_sum += jsd([ans_prob_dist[0],ans_prob_dist[1] + ans_prob_dist[2]],[pred_prob_dist[0],pred_prob_dist[1] + pred_prob_dist[2]])
            jsd_OT_X_sum += jsd([ans_prob_dist[0] + ans_prob_dist[1],ans_prob_dist[2]],[pred_prob_dist[0] + pred_prob_dist[1] ,pred_prob_dist[2]])

            mse_O_T_X_sum += mse(ans_prob_dist,pred_prob_dist)
            mse_O_TX_sum += mse([ans_prob_dist[0],ans_prob_dist[1] + ans_prob_dist[2]],[pred_prob_dist[0],pred_prob_dist[1] + pred_prob_dist[2]])
            mse_OT_X_sum += mse([ans_prob_dist[0] + ans_prob_dist[1],ans_prob_dist[2]],[pred_prob_dist[0] + pred_prob_dist[1] ,pred_prob_dist[2]])

            pred_label = target_label['breakdown']
            eval_index += 1

            if pred_label == ans_label:
                correct_num += 1
            else:
                incorrect_num += 1

            if pred_label == "O":
                if ans_label == "O":
                    predO_ansO += 1
                elif ans_label == "T":
                    predO_ansT += 1
                elif ans_label == "X":
                    predO_ansX += 1
            elif pred_label == "T":
                if ans_label == "O":
                    predT_ansO += 1
                elif ans_label == "T":
                    predT_ansT += 1
                elif ans_label == "X":
                    predT_ansX += 1
            elif pred_label == "X":
                if ans_label == "O":
                    predX_ansO += 1
                elif ans_label == "T":
                    predX_ansT += 1
                elif ans_label == "X":
                    predX_ansX += 1

            #lenient
            if pred_label == "O":
                if ans_label_l == "O_l":
                    predO_ansO_l += 1
                elif ans_label_l == "X_l":
                    predO_ansX_l += 1
            elif pred_label == "T":
                if ans_label_l == "O_l":
                    predT_ansO_l += 1
                elif ans_label_l == "X_l":
                    predT_ansX_l += 1
            elif pred_label == "X":
                if ans_label_l == "O_l":
                    predX_ansO_l += 1
                elif ans_label_l == "X_l":
                    predX_ansX_l += 1


#     print correct_num
#     print incorrect_num
#
#     print predO_ansO
#     print predO_ansT
#     print predO_ansX
#     print predT_ansO
#     print predT_ansT
#     print predT_ansX
#     print predX_ansO
#     print predX_ansT
#     print predX_ansX


    #print str(predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l + predO_ansX_l + predO_ansO_l)

    print "######### Data Stats #########"
    print "File Num : \t\t" + str(file_num)
    print "System Utterance Num : \t" + str(label_num)
    print "O Label Num : \t\t" + str(predO_ansO + predT_ansO + predX_ansO)
    print "T Label Num : \t\t" + str(predO_ansT + predT_ansT + predX_ansT)
    print "X Label Num : \t\t" + str(predO_ansX + predT_ansX + predX_ansX)
    print ""

    print "######### Results #########"
    print "Accuracy : \t\t%4f" % ((correct_num*1.0) / (correct_num + incorrect_num)) + " (" + str(correct_num) + "/" + str(correct_num + incorrect_num) + ")\n"

    precision_s = 0.0
    recall_s = 0.0
    fmeasure_s = 0.0

    if predX_ansX > 0:
        if (predX_ansO + predX_ansT + predX_ansX) > 0:
            precision_s = predX_ansX*1.0 / (predX_ansO + predX_ansT + predX_ansX)
        if (predO_ansX + predT_ansX + predX_ansX) > 0:
            recall_s = predX_ansX*1.0 / (predO_ansX + predT_ansX + predX_ansX)

    if precision_s > 0 and recall_s > 0:
        fmeasure_s = (2*precision_s*recall_s)/(precision_s + recall_s)

    print "Precision (B) : \t%4f" % (precision_s) + " (" + str(predX_ansX) + "/" + str(predX_ansO + predX_ansT + predX_ansX) + ")"
    print "Recall    (B) : \t%4f" % (recall_s) + " (" + str(predX_ansX) + "/" + str(predO_ansX + predT_ansX + predX_ansX) + ")"
    print "F-measure (B) : \t%4f" % (fmeasure_s) + "\n"


############################ PB-T,  NB-O #######################################
    precision_pb = 0.0
    recall_pb = 0.0
    fmeasure_pb = 0.0

    if predT_ansT > 0:
        if (predT_ansO + predT_ansT + predT_ansX) > 0:
            precision_pb = predT_ansT*1.0 / (predT_ansO + predT_ansT + predT_ansX)
        if (predO_ansT + predT_ansT + predX_ansT) > 0:
            recall_pb = predT_ansT*1.0 / (predO_ansT + predT_ansT + predX_ansT)

    if precision_pb > 0 and recall_pb > 0:
        fmeasure_pb = (2*precision_pb*recall_pb)/(precision_pb + recall_pb)

    print "Precision (PB) : \t%4f" % (precision_pb) + " (" + str(predT_ansT) + "/" + str(predT_ansO + predT_ansT + predT_ansX) + ")"
    print "Recall    (PB) : \t%4f" % (recall_pb) + " (" + str(predT_ansT) + "/" + str(predO_ansT + predT_ansT + predX_ansT) + ")"
    print "F-measure (PB) : \t%4f" % (fmeasure_pb) + "\n"

    precision_nb = 0.0
    recall_nb = 0.0
    fmeasure_nb = 0.0

    if predO_ansO > 0:
        if (predO_ansO + predO_ansT + predO_ansX) > 0:
            precision_nb = predO_ansO * 1.0 / (predO_ansO + predO_ansT + predO_ansX)
        if (predO_ansO + predT_ansO + predX_ansO) > 0:
            recall_nb = predO_ansO * 1.0 / (predO_ansO + predT_ansO + predX_ansO)

    if precision_nb > 0 and recall_nb > 0:
        fmeasure_nb = (2 * precision_nb * recall_nb) / (precision_nb + recall_nb)

    print "Precision (NB) : \t%4f" % (precision_nb) + " (" + str(predO_ansO) + "/" + str(predO_ansO + predO_ansT + predO_ansX) + ")"
    print "Recall    (NB) : \t%4f" % (recall_nb) + " (" + str(predO_ansO) + "/" + str(predO_ansO + predT_ansO + predX_ansO) + ")"
    print "F-measure (NB) : \t%4f" % (fmeasure_nb) + "\n"

############################ ########## #######################################

    precision_l = 0.0
    recall_l = 0.0
    fmeasure_l = 0.0
    if (predT_ansX_l + predX_ansX_l) > 0 and (predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l) > 0:
        precision_l = (predT_ansX_l + predX_ansX_l)*1.0 / (predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l)
    if (predT_ansX_l + predX_ansX_l) > 0 and (predO_ansX_l + predT_ansX_l + predX_ansX_l) > 0:
        recall_l = (predT_ansX_l + predX_ansX_l)*1.0 / (predO_ansX_l + predT_ansX_l + predX_ansX_l)
    if precision_l > 0 and recall_l > 0:
        fmeasure_l = (2*precision_l*recall_l)/(precision_l + recall_l)



    print "Precision (PB+B) : \t%4f" % (precision_l) + " (" + str(predT_ansX_l + predX_ansX_l) + "/" + str(predX_ansO_l + predX_ansX_l + predT_ansO_l + predT_ansX_l) + ")"
    print "Recall    (PB+B) : \t%4f" % (recall_l) + " (" + str(predT_ansX_l + predX_ansX_l) + "/" + str(predO_ansX_l + predT_ansX_l + predX_ansX_l) + ")"
    print "F-measure (PB+B) : \t%4f" % (fmeasure_l) + "\n"

    print "JS divergence (NB,PB,B) : \t%4f" % (jsd_O_T_X_sum / (correct_num + incorrect_num))
    print "JS divergence (NB,PB+B) : \t%4f" % (jsd_O_TX_sum / (correct_num + incorrect_num))
    print "JS divergence (NB+PB,B) : \t%4f" % (jsd_OT_X_sum / (correct_num + incorrect_num)) + "\n"

    print "Mean squared error (NB,PB,B) : \t%4f" % (mse_O_T_X_sum / (correct_num + incorrect_num))
    print "Mean squared error (NB,PB+B) : \t%4f" % (mse_O_TX_sum / (correct_num + incorrect_num))
    print "Mean squared error (NB+PB,B) : \t%4f" % (mse_OT_X_sum / (correct_num + incorrect_num))
    print "###########################"


    print "********************** TABULATE **********************"

    print eval_path
    print "CLS : \t%4f\t%4f\t%4f" % ((correct_num*1.0) / (correct_num + incorrect_num), fmeasure_s, fmeasure_l)
    print "JSD : \t%4f\t%4f\t%4f" % (jsd_O_T_X_sum / (correct_num + incorrect_num), jsd_O_TX_sum / (correct_num + incorrect_num), jsd_OT_X_sum / (correct_num + incorrect_num))
    print "MSE : \t%4f\t%4f\t%4f" % (mse_O_T_X_sum / (correct_num + incorrect_num), mse_O_TX_sum / (correct_num + incorrect_num), mse_OT_X_sum / (correct_num + incorrect_num))


if __name__ == '__main__':
    main()
