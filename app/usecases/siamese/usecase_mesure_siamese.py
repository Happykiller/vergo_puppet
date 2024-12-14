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
        similarity_precision = []
        details = []
        
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

            # Record similarity precision as a percentage
            precision = (1 - error) * 100  # Precision as 100% when no error
            similarity_precision.append(precision)
            
            # Check if prediction is correct
            is_correct = error <= 0.1
            if is_correct:
                correct_predictions += 1
            
            # Store detailed results for each test case
            details.append({
                "vector1": vector1,
                "vector2": vector2,
                "expected_similarity": expected_similarity,
                "predicted_similarity": predicted_similarity,
                "error": error,
                "precision_percentage": precision,
                "is_correct": is_correct
            })
            
            # Log the output details
            logger.info(f"Query: {vector1}, Image: {vector2}")
            logger.info(f"Expected similarity: {expected_similarity*100}%, Model similarity: {predicted_similarity*100:.2f}%, Error: {error*100:.2f}%")
        
        # Calculate metrics
        prediction_accuracy = (correct_predictions / total_tests) * 100
        avg_similarity_precision = sum(similarity_precision) / total_tests

        # Compile the report
        report = {
            "model_name": name,
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "prediction_accuracy_percentage": prediction_accuracy,
            "avg_similarity_precision_percentage": avg_similarity_precision,
            "details": details
        }
        
        # Log summary details
        logger.info("---------------------")
        logger.info(f"Prediction accuracy: {prediction_accuracy:.2f}% ({correct_predictions}/{total_tests})")
        logger.info(f"Average similarity precision: {avg_similarity_precision:.2f}%")


        return report
    
    except Exception as e:
        # General error handling
        logger.error(f"An error occurred during siamese testing: {str(e)}")
        raise Exception(f"An error occurred during siamese testing: {str(e)}")
