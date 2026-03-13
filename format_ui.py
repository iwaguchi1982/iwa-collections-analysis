import os

def wrap_1gene():
    path = "ui/analysis_1gene.py"
    with open(path, "r") as f:
        lines = f.read().split("\n")
    
    start_idx = -1
    for i, l in enumerate(lines):
        if 'st.subheader("Statistical Significance & Risk Analysis")' in l:
            start_idx = i - 1  # get the ---
            break
            
    if start_idx == -1: return

    out = lines[:start_idx]
    
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    sp = " " * base_indent
    
    out.append(f'{sp}# Wrap remainder of page in expander')
    out.append(f'{sp}with st.expander("📂 Show Detailed Statistical & Clinical Analysis", expanded=False):')
    
    for l in lines[start_idx:]:
        if len(l.strip()) == 0:
            out.append("")
        else:
            out.append("    " + l)
            
    with open(path, "w") as f:
        f.write("\n".join(out))
    print("Wrapped 1-gene")

def wrap_2gene():
    path = "ui/analysis_2gene.py"
    with open(path, "r") as f:
        lines = f.read().split("\n")
    
    start_idx = -1
    for i, l in enumerate(lines):
        if 'st.subheader("Group Background Check")' in l:
            start_idx = i - 1  # get the ---
            break
            
    if start_idx == -1: return

    out = lines[:start_idx]
    
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    sp = " " * base_indent
    
    out.append(f'{sp}# Wrap remainder of page in expander')
    out.append(f'{sp}with st.expander("📂 Show Detailed Statistical & Clinical Analysis", expanded=False):')
    
    for l in lines[start_idx:]:
        if len(l.strip()) == 0:
            out.append("")
        else:
            out.append("    " + l)
            
    with open(path, "w") as f:
        f.write("\n".join(out))
    print("Wrapped 2-gene")

wrap_1gene()
wrap_2gene()
