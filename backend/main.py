from collections import Counter

def majority_vote(window_predictions):
    """
    Determines the final prediction for an audio file based on majority vote
    from its window predictions.

    Args:
        window_predictions (list): A list of prediction labels (e.g., strings
                                   representing classes like 'AS', 'MR', 'N')
                                   for all windows of a single audio file.

    Returns:
        str: The final predicted class label for the audio file.
             Returns None if window_predictions is empty.
             If there's a tie, it can be handled (e.g., by returning one of
             the tied classes or a specific "tie" indicator). Currently,
             `Counter.most_common(1)` will return one of the most common items
             in case of a tie.
    """
    if not window_predictions:
        return None  # Or raise an error, or return a default class

    vote_counts = Counter(window_predictions)
    # most_common(1) returns a list of tuples [(item, count)], so get the item
    majority_class, _ = vote_counts.most_common(1)[0]

    return majority_class

if __name__ == '__main__':
    # Example Usage
    print("Testing majority_vote function...")

    # Example 1: Clear majority
    predictions1 = ['AS', 'AS', 'N', 'AS', 'MR', 'AS']
    final_prediction1 = majority_vote(predictions1)
    print(f"Window predictions: {predictions1}, Final prediction: {final_prediction1}") # Expected: AS

    # Example 2: Tie case (Counter.most_common(1) behavior)
    predictions2 = ['AS', 'AS', 'N', 'N', 'MR']
    final_prediction2 = majority_vote(predictions2)
    print(f"Window predictions: {predictions2}, Final prediction: {final_prediction2}") # Expected: AS or N (depends on internal tie-breaking of Counter)

    # Example 3: All same
    predictions3 = ['MR', 'MR', 'MR', 'MR']
    final_prediction3 = majority_vote(predictions3)
    print(f"Window predictions: {predictions3}, Final prediction: {final_prediction3}") # Expected: MR

    # Example 4: Empty list
    predictions4 = []
    final_prediction4 = majority_vote(predictions4)
    print(f"Window predictions: {predictions4}, Final prediction: {final_prediction4}") # Expected: None

    # Example 5: Single prediction
    predictions5 = ['N']
    final_prediction5 = majority_vote(predictions5)
    print(f"Window predictions: {predictions5}, Final prediction: {final_prediction5}") # Expected: N
