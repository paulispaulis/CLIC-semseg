"""
Automatic latex generation scripts.
"""

import numpy as np

def make_table(array, x_labels, y_labels):
    """
    Generate latex code for a table.
    array - array of shape [n, m] with percentages.
    x_labels - list of length [m] with labels for x axis.
    y_labels - list of length [n] with labels for y axis.
    Returns string.
    
    You have to add usepackage{booktabs} to your latex file
    for table to format correctly.
    """
    if array.shape != (len(y_labels), len(x_labels)):
        return "Dimension mismatch between array and labels."
    
    n, m = array.shape
    latex_table = []
    
    # Begin LaTeX table
    latex_table.append("\\begin{tabular}{l|" + "c" * m + "}")
    latex_table.append("\\toprule")
    
    # Add x labels
    latex_table.append(" & " + " & ".join(x_labels) + " \\\\")
    latex_table.append("\\midrule")
    
    # Add rows
    for i in range(n):
        row = [str(y_labels[i])] + [str(array[i, j]) for j in range(m)]
        latex_table.append(" & ".join(row) + " \\\\")
        
    # End LaTeX table
    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabular}")
    
    return "\n".join(latex_table)