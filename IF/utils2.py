from sklearn.preprocessing import MinMaxScaler
import numpy as np


def input_data(df, test_idx, Y_test):
    scaler = MinMaxScaler()

    if sum(df.Influence.tolist()) == 0:
        df = df[df.Similarity > 0.3].sort_values('Similarity', ascending=False)  # .reset_index(drop=True)
        df_pos = df[df.Y_train == Y_test[test_idx].item()]
        df_neg = df[df.Y_train != Y_test[test_idx].item()]
    else:
        df_pos = df[df.Influence > 0].sort_values('Influence', ascending=False)  # .reset_index(drop=True)
        df_pos = df_pos[df_pos.Y_train == Y_test[test_idx].item()]
        df_pos = df_pos.sort_values('Similarity', ascending=False)[:200]
        df_pos = df_pos.sort_values('Influence', ascending=False)[:150]
        df_pos[['Influence', 'Similarity']] = scaler.fit_transform(df_pos[['Influence', 'Similarity']])

        df_neg = df[df.Influence < 0].sort_values('Influence', ascending=True)  # .reset_index(drop=True)
        df_neg = df_neg[df_neg.Y_train != Y_test[test_idx].item()]  # [:nn]
        df_neg = df_neg.sort_values('Similarity', ascending=False)[:20]
        df_neg = df_neg.sort_values('Influence', ascending=True)[:10]
        df_neg[['Influence', 'Similarity']] = scaler.fit_transform(df_neg[['Influence', 'Similarity']])

    return df_pos, df_neg


def calculate_cosine_similarity(a, b):
    """Calculate cosine similarity between two arrays."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def greedy_subset_selection(df, N, sett=None):
    arrays = [np.array(i) for i in df.X_train]
    influence_scores = df.Influence.tolist()  # List of influence scores for each array
    prox = df.Similarity.tolist()
    #     infsim = df.infsim.tolist()
    n_arrays = len(arrays)
    selected_indices = []
    # Start with the array with the highest influence score

    if sett == 'positive':
        initial_idx = np.argmax(influence_scores)
    else:
        initial_idx = np.argmin(influence_scores)

    selected_indices.append(initial_idx)
    selected_array = arrays[initial_idx]

    while len(selected_indices) < N:
        max_gain = -np.inf
        selected_idx = None

        # Iterate over the remaining arrays
        for i in range(n_arrays):
            if i not in selected_indices:
                current_array = arrays[i]
                final_list = list(
                    map(lambda x: calculate_cosine_similarity(current_array, arrays[x]), selected_indices))
                if any(i > 0.9 for i in final_list):
                    continue
                else:
                    if sett == 'positive':
                        combined_score = influence_scores[i] + prox[i] / 5  # -0.4*similarity
                    else:
                        combined_score = -0.1 * influence_scores[i] + prox[i]

                    if combined_score > max_gain:
                        max_gain = combined_score
                        selected_idx = i

        # Add selected array to the subset
        selected_indices.append(selected_idx)

    return selected_indices