import os
import numpy as np
import pandas as pd


def compute_metrics(
    datasets_orig,
    datasets_rec,
    titles,
    output_dir
):
    results = {'title': [], 'rec_error': []}
    for i in range(len(datasets_orig)):
        X_orig, _ = datasets_orig[i]
        X_rec, _ = datasets_rec[i]
        # Reconstruction error
        rec_error = np.linalg.norm((X_orig - X_rec).numpy().flatten())

        results['title'].append(titles[i])
        results['rec_error'].append(rec_error)
    
    # Save the results to a .txt file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'metrics.txt'), sep='\t', index=False)
