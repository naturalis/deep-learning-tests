text_file = open("sci_names.txt", "r")
lines = text_file.readlines()
text_file.close()
print(len(lines))

uniq = list(set(lines))
print(len(uniq))

with open("sci_names_uniq.txt", 'w') as f:
    for item in uniq:
        f.write("%s\n" % item)
