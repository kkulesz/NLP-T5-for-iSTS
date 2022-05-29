# encoding: utf-8

import logging
import re
import sys
import consts
import os

sys.path.append('engine')

test_data_path = os.path.join(consts.current_dataset, consts.test_data_file_name)

def create_wa():
    csvfile = open("out.csv")
    csvfile.readline() #pominiecie pierwszej linii


    alignment_start_pattern = re.compile(r"<alignment>")
    alignment_end_pattern = re.compile(r"</alignment>")
    equality_pattern = re.compile(r"<==>")
    doubleslash_pattern = re.compile(r"//")

    logger = logging.getLogger("model.wa")
    logger.info("Start inferencing and creating .wa files")

    aligment_section = False
    class_dict = { 0 : 'OPPO', 1 : 'REL', 2 : 'SIMI', 3 : 'SPE1',4 : 'SPE2', 5: 'EQUI'}
    processed_file = []
    processed_file.append('<sentence id="1" status="">\n')
    file_name = "out_predicted.wa"
    with open("data/answers-students/test_gs.wa", 'rb') as fin:
        fin.readline()
        for index, line in enumerate(fin):

            processed_line = []
            line = line.decode('latin-1')
            if alignment_end_pattern.match(line):
                aligment_section = False
            if aligment_section:
                equality_pos = [m.start() for m in re.finditer(equality_pattern, line)]
                doubleslash_pos = [m.start() for m in re.finditer(doubleslash_pattern, line)]

                alignment = line[:doubleslash_pos[0]]
                value = line[doubleslash_pos[1] + 2: doubleslash_pos[2]]
                explanation = line[doubleslash_pos[0] + 2: doubleslash_pos[1]]
                first_chunk = line[doubleslash_pos[2] + 2: equality_pos[1]]
                second_chunk = line[equality_pos[1] + 4:]

                if value != " NIL " and explanation != ' ALIC ' and explanation != ' NOALI ':
                    split = csvfile.readline().split(',')
                    type = int(split[0].strip('\n\r'))
                    newval = split[1].strip('\n\r')

                    if explanation.strip() == "EQUI":
                        newval = " 5 "
                    explanation = " " + class_dict[type] + " "

                processed_line = alignment + "//" + explanation + "// " + newval + " //" + first_chunk + "<==>" + second_chunk
                processed_file.append(processed_line)
            else:
                processed_file.append(line)

            if alignment_start_pattern.match(line):
                aligment_section = True

    with open(file_name, "w", newline='', encoding="utf-8") as f:
        for item in processed_file:
            f.write("%s" % item)


def main():
    create_wa()

if __name__ == '__main__':
    main()