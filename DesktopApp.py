

import tkinter as tk
from tkinter import filedialog
import csv

window = tk.Tk()
window.title("ELEC390")
window.geometry("400x400")

label = tk.Label(window, text="Input a file in CSV format:")
label.pack()

text = tk.Text(window)
text.pack()


def open_file() :
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        with open(file_path, "r") as f:
            contents = f.read()
            text.insert(tk.END, contents)

            rows = contents.split("/n")
            processed_data = []
            for row in rows:
                processed_data = row.split(",")
                processed_data.append(processed_data)

            output_path = filedialog.asksaveasfilename(defaultextension=".csv")
            if output_path:
                with open(output_path, "w", newline="") as output_file:
                    write =  csv.writer(output_file)
                    write.writerows(processed_data)


button = tk.Button(window, text="Open File", command=open_file)
button.pack()

window.mainloop()



