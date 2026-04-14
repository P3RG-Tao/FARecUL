# d.py
import csv

class DataReorganizer:
    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename

    def reorganize_data(self):
        max_value = 0
        total_rows = 0
        with open(self.input_filename, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader)  # 跳过第一行（标题行）
            rows = list(reader)
            total_rows = len(rows)
            max_value = max(int(row[0]) for row in rows if row[0].isdigit())

        with open(self.output_filename, mode='w', encoding='utf-8') as outfile:
            for row in rows:
                row[1] = str(int(row[1]) + max_value + 1)
                outfile.write(' '.join(row) + '\n')

        return max_value, total_rows

    def run(self):
        return self.reorganize_data()