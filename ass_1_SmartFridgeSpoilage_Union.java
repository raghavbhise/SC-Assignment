import java.util.Scanner;

public class ass_1_SmartFridgeSpoilage_Union {

    // -------------------------
    // Membership Functions
    // -------------------------

    // Fuzzify temperature (-10 to 20 °C) to membership value between 0 and 1
    static double fuzzifyTemperature(double temp) {
        return Math.max(0.0, Math.min(1.0, (temp + 10) / 30.0));
    }

    // Fuzzify moisture (0 to 100%) to membership value between 0 and 1
    static double fuzzifyMoisture(double moisture) {
        return Math.max(0.0, Math.min(1.0, moisture / 100.0));
    }

    // Fuzzify number of days (0 to 5) to membership value between 0 and 1
    static double fuzzifyDays(double days) {
        return Math.max(0.0, Math.min(1.0, days / 5.0));
    }

    // -------------------------
    // Main Program
    // -------------------------
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        // Take user input for temperature
        System.out.print("Enter Temperature (-10 to 20 °C): ");
        double temp = scanner.nextDouble();

        // Take user input for moisture level
        System.out.print("Enter Moisture (0 to 100%): ");
        double moisture = scanner.nextDouble();

        // Take user input for number of days in fridge
        System.out.print("Enter Days in fridge (0 to 5): ");
        double days = scanner.nextDouble();

        // Fuzzify each input to get membership values
        double tempMF = fuzzifyTemperature(temp);
        double moistureMF = fuzzifyMoisture(moisture);
        double daysMF = fuzzifyDays(days);

        // Combine all fuzzy values using UNION (maximum of the three)
        double spoilageIndex = Math.max(tempMF, Math.max(moistureMF, daysMF));

        // -------------------------
        // Display Results
        // -------------------------
        System.out.println("\n--- Fuzzy Spoilage Detection (Union) ---");
        System.out.printf("Temperature MF: %.2f\n", tempMF);
        System.out.printf("Moisture MF: %.2f\n", moistureMF);
        System.out.printf("Days MF: %.2f\n", daysMF);
        System.out.printf("Spoilage Index (Union): %.2f\n", spoilageIndex);

        // Determine spoilage risk level based on spoilage index
        if (spoilageIndex >= 0.7) {
            System.out.println("⚠️ Spoilage Risk: HIGH");
        } else if (spoilageIndex >= 0.4) {
            System.out.println("⚠️ Spoilage Risk: MEDIUM");
        } else {
            System.out.println("✅ Spoilage Risk: LOW");
        }

        // Close scanner to avoid resource leak
        scanner.close();
    }
}
