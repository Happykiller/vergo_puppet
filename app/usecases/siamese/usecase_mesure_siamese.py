#app\usecases\siamese\usecase_mesure_siamese.py
from app.services.logger import logger
from app.repositories.memory import get_model
from app.neural_network.nn_siamese import evaluate_similarity
from app.usecases.siamese.usecase_commons_siamese import create_indexed_glossary, tokens_to_indices

def mesure_siamese(name, test_data):
    try:
        total_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        detailed_results = []
        
        # Retrieve the model
        model = get_model(name)
        if not model:
            raise Exception("Model not found")

        glossary = model.get("glossary", [])
        nn_model = model.get("nn_model", None)
        if not nn_model:
            raise Exception("Model not completed")

        # Create index mapping for glossary terms
        word2idx = create_indexed_glossary(glossary)
        
        # Iterate through test data
        for vector1, vector2, expected_similarity in test_data:
            # Convert token lists to indices
            vector1_indices = tokens_to_indices(vector1, word2idx)
            vector2_indices = tokens_to_indices(vector2, word2idx)
            
            # Evaluate similarity
            predicted_similarity = evaluate_similarity(nn_model, vector1_indices, vector2_indices)
            error = abs(predicted_similarity - expected_similarity)
            total_error += error
            
            # Consider the prediction correct if the error is below a threshold (e.g., 0.1)
            is_correct = error <= 0.1
            if is_correct:
                correct_predictions += 1

            # Add details of the current test case to the results
            detailed_results.append({
                "query": vector1,
                "image": vector2,
                "expected_similarity": expected_similarity,
                "predicted_similarity": predicted_similarity,
                "error": error,
                "is_correct": is_correct,
            })
            
            # Log the output details
            logger.info(f"Query: {vector1}, Image: {vector2}")
            logger.info(f"Expected similarity: {expected_similarity*100}%, Model similarity: {predicted_similarity*100:.2f}%, Error: {error*100:.2f}%")
        
        # Calculate average error and accuracy as a percentage
        avg_error = total_error / total_tests
        precision_percentage = (1 - avg_error) * 100  # Lower avg_error corresponds to higher accuracy

        # Generate final report
        report = {
            "model_name": name,
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "accuracy_percentage": precision_percentage,
            "average_error": avg_error,
            "detailed_results": detailed_results,
        }
        
        # Log the number of correct predictions out of the total test cases
        logger.info(f"Correct predictions: {correct_predictions}/{total_tests}")
        logger.info(f"Model average accuracy on the test set: {precision_percentage:.2f}%")

        return report
    
    except Exception as e:
        # General error handling
        logger.error(f"An error occurred during siamese testing: {str(e)}")
        raise Exception(f"An error occurred during siamese testing: {str(e)}")
