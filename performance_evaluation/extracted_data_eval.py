import json
from Levenshtein import ratio

# script compares two JSON files containing person data or raw transcription:
# - 1 file has manually transcribed (truth) data
# - 1 file has LLM-generated (test) data
#
# Key notes:
# 1. there is not a fixed number of fields
# 2. We have a separate metric for "null false positives" to avoid
#    skewing the overall accuracy metrics

# Compares two strings and sees if they're similar enough based on threshold
# Returns True if similarity is above threshold (80% - can be changed)
def string_compare(val1, val2, thresh=0.8):
    if not isinstance(val1, str) or not isinstance(val2, str):
        return False
    return ratio(val1, val2) >= thresh

# General purpose comparison function for any type of value
# Handles strings, booleans, numbers differently
def compare_vals(truth_val, test_val, thresh=0.8):
    # Cant compare if either value is missing
    if truth_val is None or test_val is None:
        return False

    # For strings, use Levenshtein matching function
    if isinstance(truth_val, str) and isinstance(test_val, str):
        return string_compare(truth_val, test_val, thresh)

    # Booleans (True/False) need exact matches
    if isinstance(truth_val, bool) and isinstance(test_val, bool):
        return truth_val == test_val

    # For everything else, try exact matching
    # We use try/except because comparing more complex types might fail
    try:
        return truth_val == test_val
    except:
        return False

# Checks if a value should be considered empty
# important for the null false positive detection
# Returns True for:
# None values and Empty lists
def is_empty_or_null(val):
    if val is None:
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    return False

# calculating metrics for a single field
# two types of comparisons:
# 1. Regular metrics for non-null fields
# 2. Special tracking of null false positives
def get_field_metrics(truth_person, test_person, field_name, threshold=0.8):
    # Check if field exists in both records
    has_truth = field_name in truth_person
    has_test = field_name in test_person

    # Initialize scoring variables
    # Regular metrics:
    true_pos = 0  # Correct matches
    false_pos = 0  # Wrong values in test data
    false_neg = 0  # Missing values that should be there

    # Special metric for null/empty list false positives
    # catches when LLM generates content where it should be null
    null_false_positives = 0

    # If both records have this field - compare
    if has_truth and has_test:
        truth_val = truth_person[field_name]
        test_val = test_person[field_name]

        # null false positive check:
        # If truth is null/empty but test isnt, that is the problem
        if is_empty_or_null(truth_val) and not is_empty_or_null(test_val):
            null_false_positives = 1
            # We return early with zeroed regular metrics
            # because this is a special case
            return {
                'true_pos': 0,
                'false_pos': 0,
                'false_neg': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'null_false_positives': 1
            }

        # For strings, we can have partial matches
        if isinstance(truth_val, str) and isinstance(test_val, str):
            sim = ratio(truth_val, test_val)
            true_pos = sim
            false_pos = 1-sim
            false_neg = 1-sim
        else:
            # For non-strings, it's a match or not
            if truth_val == test_val:
                true_pos = 1  # Perfect match
                false_pos = 0
                false_neg = 0
            else:
                true_pos = 0  # Mismatch
                false_pos = 1
                false_neg = 1
    elif has_test:
        # If only test has the field, check if it's a null false positive
        test_val = test_person[field_name]
        if not is_empty_or_null(test_val):
            null_false_positives = 1
        false_pos = 1
    elif has_truth:
        false_neg = 1

    # Calculate our accuracy scores
    # Precision = how many of our matches were correct
    # Recall = how many correct values did we find
    # F1 = balanced score between precision and recall
    prec = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    rec = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_score = 2*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0

    return {
        'true_pos': true_pos,
        'false_pos': false_pos,
        'false_neg': false_neg,
        'precision': prec,
        'recall': rec,
        'f1': f1_score,
        'null_false_positives': null_false_positives
    }

# compares entire person records
# It keeps regular metrics and null false positives separate
# so they don't interfere with each other
def compare_people(truth_person, test_person, thresh=0.8):
    # Get all fields that appear in either record
    fields = set(truth_person.keys()) | set(test_person.keys())

    # Metrics both per-field and overall
    field_metrics = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_null_fps = 0  # counter tracking null false positives

    # Calculate metrics for each field
    for f in fields:
        metrics = get_field_metrics(truth_person, test_person, f, thresh)
        field_metrics[f] = metrics

        # totals
        total_tp += metrics['true_pos']
        total_fp += metrics['false_pos']
        total_fn += metrics['false_neg']
        total_null_fps += metrics['null_false_positives']

    # Calculate overall scores from the totals
    overall_prec = (total_tp/(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0)
    overall_rec = (total_tp/(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0)
    f1 = 2*(overall_prec*overall_rec)/(overall_prec+overall_rec) if (overall_prec+overall_rec) > 0 else 0

    return {
        'field_metrics': field_metrics,
        'overall': {
            'precision': overall_prec,
            'recall': overall_rec,
            'f1_score': f1,
            'true_pos': total_tp,
            'false_pos': total_fp,
            'false_neg': total_fn,
            'null_false_positives': total_null_fps
        }
    }

def compare_raw_texts(truth_entries, test_entries):
    # Create dictionaries of entries (index by ID)
    truth_texts = {entry['id']: entry['raw'] for entry in truth_entries}
    test_texts = {entry['id']: entry['raw'] for entry in test_entries}

    # Find common IDs between both files
    common_ids = set(truth_texts.keys()) & set(test_texts.keys())

    metrics = {
        'true_pos': 0,
        'false_pos': 0,
        'false_neg': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'entries_compared': len(common_ids),
        'entry_details': []
    }

    # Compare each matching entry
    for entry_id in common_ids:
        truth_text = truth_texts[entry_id]
        test_text = test_texts[entry_id]

        similarity = ratio(truth_text, test_text)
        metrics['true_pos'] += similarity
        metrics['false_pos'] += (1 - similarity)
        metrics['false_neg'] += (1 - similarity)

        # Store details for this entry
        metrics['entry_details'].append({
            'id': entry_id,
            'similarity': similarity
        })

    # Calculate overall metrics if we had any entries to compare
    if metrics['entries_compared'] > 0:
        metrics['precision'] = metrics['true_pos'] / (metrics['true_pos'] + metrics['false_pos'])
        metrics['recall'] = metrics['true_pos'] / (metrics['true_pos'] + metrics['false_neg'])
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])

    return metrics

def compare_people_data(truth_entries, test_entries):
    # JSON format
    truth_person = truth_entries[0]['data']['people'][0]
    test_person = test_entries[0]['data']['people'][0]

    results = compare_people(truth_person, test_person)
    return results

def print_raw_metrics(metrics):
    print("\nRaw Text Comparison Results:")
    print(f"Number of entries compared: {metrics['entries_compared']}")

    # Print details for each entry
    print("\nIndividual Entry Results:")
    for entry in metrics['entry_details']:
        print(f"Entry ID: {entry['id']}")
        similarity = entry['similarity']

        # Calculate individual entry metrics (same calculations as above - kind of redundant!)
        precision = similarity / (similarity + (1-similarity))
        recall = similarity / (similarity + (1-similarity))
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 30)

    print("\nOverall Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

def print_people_metrics(results):
    # Print results for each field
    print("\nPeople Data Comparison Results:")
    print("\nMetrics by field:")
    for field, m in results['field_metrics'].items():
        print(f"\n{field}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall: {m['recall']:.4f}")
        print(f"  F1: {m['f1']:.4f}")
        # Only show null false positives if we found any
        if m['null_false_positives'] > 0:
            print(f"  Null False Positives: {m['null_false_positives']}")

    print("\nOverall scores:")
    overall = results['overall']
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall: {overall['recall']:.4f}")
    print(f"F1: {overall['f1_score']:.4f}")
    print(f"Total Null False Positives: {overall['null_false_positives']}")

# The main function handles file reading and result printing
def main():
    # Get file paths
    truth_file = input("Enter truth path (human-generated) JSON file: ")
    test_file = input("Enter test path (LLM-generated) JSON file: ")

    # Load JSON files
    try:
        with open(truth_file, 'r') as f:
            truth = json.load(f)
        with open(test_file, 'r') as f:
            test = json.load(f)
    except FileNotFoundError:
        print("Error: One or both files not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in one or both files")
        return

    while True:
        print("\nWhat would you like to compare?")
        print("1. Raw transcription text")
        print("2. People data")
        print("3. Exit")

        choice = input("\nEnter (1-3): ")

        if choice == "1":
            metrics = compare_raw_texts(truth['entries'], test['entries'])
            # Print overall scores
            print_raw_metrics(metrics)
        elif choice == "2":
            # Get the first person from each file
            # assuming the specific JSON structure:
            # {"entries": [{"data": {"people": [...]}}]}
            truth_person = truth['entries'][0]['data']['people'][0]
            test_person = test['entries'][0]['data']['people'][0]
            # Run comparison and get all metrics
            results = compare_people(truth_person, test_person)
            # Print overall scores
            print_people_metrics(results)
        elif choice == "3":
            break
        else:
            print("Invalid option. Enter 1, 2, or 3.")

if __name__ == "__main__":
    main()