# Define input and output file paths
input_file = 'german.txt'
output_file = 'words.txt'

chars = ['a', 'ä', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'q', 'r', 's', 't', 'u', 'ü', 'v', 'w', 'x', 'y', 'z']

# Open the input file for reading and output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Iterate through each line in the input file
    for line in infile:
        # Split the line by tab characters
        columns = line.split('\t')
        # Check if there are at least 2 columns
        if len(columns) >= 2:
            # Extract the second column (word) and write it to the output file
            word = columns[1].lower()
            if all(char in chars for char in word):
            	outfile.write(word + '\n')
