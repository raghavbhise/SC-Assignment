import java.util.Scanner; // Import Scanner class for taking user input

public class ass_2_FuzzyWorkoutRecommender {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in); // Scanner object to read input

        // 🔹 Fuzzy relation matrix (Exercise Suitability)
        // Each row represents an exercise
        // Each column represents how well the exercise improves a fitness goal:
        // [Strength, Stamina, Flexibility]
        double[][] relation = {
                { 0.9, 0.4, 0.2 }, // Weightlifting is very good for Strength, moderate for Stamina, poor for
                                   // Flexibility
                { 0.3, 0.9, 0.3 }, // Running is very good for Stamina, weak for Strength and Flexibility
                { 0.2, 0.3, 0.9 }, // Yoga is excellent for Flexibility, poor for Strength and Stamina
                { 0.4, 0.8, 0.4 }, // Cycling is strong for Stamina, moderate for Strength and Flexibility
                { 0.8, 0.5, 0.3 }, // Push-ups are strong for Strength, decent for Stamina, weak for Flexibility
                { 0.2, 0.2, 0.8 } // Stretching is excellent for Flexibility, weak for Strength and Stamina
        };

        // Exercise and Goal names for printing
        String[] exercises = { "Weightlifting", "Running", "Yoga", "Cycling", "Push-ups", "Stretching" };
        String[] goals = { "Strength", "Stamina", "Flexibility" };

        // 🔹 Take user input in percentage (0–100) for each fitness goal
        double[] userGoals = new double[3];
        for (int i = 0; i < goals.length; i++) {
            System.out.print("Enter your " + goals[i] + " level (% 0-100): ");
            userGoals[i] = sc.nextDouble() / 100.0; // Normalize input from [0–100] to [0–1]
        }

        // 🔹 Calculate deficiency = 1 - input
        // Example: If user has only 30% Strength → 0.3 → deficiency = 0.7 (needs more
        // strength training)
        double[] deficiency = new double[3];
        for (int i = 0; i < goals.length; i++) {
            deficiency[i] = 1 - userGoals[i];
        }

        // 🔹 Calculate recommendation scores using fuzzy relation
        // Formula: ExerciseScore = max(min(deficiency[j], relation[i][j]))
        double[] exerciseScores = new double[exercises.length];
        for (int i = 0; i < exercises.length; i++) {
            double score = 0.0;
            for (int j = 0; j < goals.length; j++) {
                // Fuzzy AND → min(deficiency, relation)
                // Fuzzy OR → max across all goals
                score = Math.max(score, Math.min(deficiency[j], relation[i][j]));
            }
            exerciseScores[i] = score; // Final score for each exercise
        }

        // 🔹 Show detailed results
        System.out.println("\n--- Exercise Scores ---");
        for (int i = 0; i < exercises.length; i++) {
            System.out.printf("%s: %.2f%n", exercises[i], exerciseScores[i]);
        }

        // 🔹 Give recommendations based on scores
        System.out.println("\n--- Conclusion ---");
        System.out.println("Recommended Exercises:");
        for (int i = 0; i < exercises.length; i++) {
            if (exerciseScores[i] >= 0.7) { // Very strong recommendation
                System.out.println("Highly Recommended: " + exercises[i]);
            } else if (exerciseScores[i] >= 0.4) { // Moderate recommendation
                System.out.println("Suggested: " + exercises[i]);
            }
        }
    }
}
