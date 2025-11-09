import java.util.Scanner;

public class ass_3_SmartFridgeSpoilage_10mf {

    // ---------- MEMBERSHIP FUNCTIONS ----------

    // 1. Triangular MF
    static double triangularMF(double x, double a, double b, double c) {
        if (x <= a || x >= c)
            return 0.0;
        else if (x == b)
            return 1.0;
        else if (x > a && x < b)
            return (x - a) / (b - a);
        else
            return (c - x) / (c - b);
    }

    // 2. Trapezoidal MF
    static double trapezoidalMF(double x, double a, double b, double c, double d) {
        if (x <= a || x >= d)
            return 0.0;
        else if (x >= b && x <= c)
            return 1.0;
        else if (x > a && x < b)
            return (x - a) / (b - a);
        else
            return (d - x) / (d - c);
    }

    // 3. Gaussian MF
    static double gaussianMF(double x, double c, double sigma) {
        return Math.exp(-0.5 * Math.pow((x - c) / sigma, 2));
    }

    // 4. Generalized Bell MF
    static double bellMF(double x, double a, double b, double c) {
        return 1.0 / (1 + Math.pow(Math.abs((x - c) / a), 2 * b));
    }

    // 5. Sigmoidal MF
    static double sigmoidMF(double x, double a, double c) {
        return 1.0 / (1 + Math.exp(-a * (x - c)));
    }

    // 6. Left-Right MF
    static double leftRightMF(double x, double a, double b, double c) {
        if (x < b)
            return Math.exp(-Math.pow((x - b) / a, 2));
        else
            return 1.0 / (1 + Math.pow((x - b) / c, 2));
    }

    // 7. Π-shaped (Pi-shaped) MF
    static double piMF(double x, double a, double b, double c, double d) {
        if (x <= a || x >= d)
            return 0.0;
        else if (x >= b && x <= c)
            return 1.0;
        else if (x > a && x < b)
            return 2 * Math.pow((x - a) / (b - a), 2);
        else
            return 1 - 2 * Math.pow((x - c) / (d - c), 2);
    }

    // 8. Open Left MF
    static double openLeftMF(double x, double a, double b) {
        if (x <= a)
            return 1.0;
        else if (x >= b)
            return 0.0;
        else
            return (b - x) / (b - a);
    }

    // 9. Open Right MF
    static double openRightMF(double x, double a, double b) {
        if (x >= b)
            return 1.0;
        else if (x <= a)
            return 0.0;
        else
            return (x - a) / (b - a);
    }

    // 10. S-shaped MF
    static double sShapedMF(double x, double a, double b) {
        if (x <= a)
            return 0.0;
        else if (x >= b)
            return 1.0;
        else {
            double t = (x - a) / (b - a);
            if (t <= 0.5)
                return 2 * t * t;
            else
                return 1 - 2 * Math.pow(1 - t, 2);
        }
    }

    // ---------- FUZZY OPERATION ----------
    static double fuzzySpoilage(double moisture, double days, double temp, int choice) {
        double mVal = 0, dVal = 0, tVal = 0;

        // --- Normalization ---
        double mNorm = moisture / 100.0; // Moisture: 0–100
        double dNorm = days / 10.0; // Days: 0–10
        double tNorm = temp / 20.0; // Temp: 0–20 °C

        switch (choice) {
            case 1: // Triangular
                mVal = triangularMF(mNorm, 0.2, 0.5, 0.8);
                dVal = triangularMF(dNorm, 0.1, 0.5, 1.0);
                tVal = triangularMF(tNorm, 0.0, 0.5, 1.0);
                break;
            case 2: // Trapezoidal
                mVal = trapezoidalMF(mNorm, 0.2, 0.4, 0.6, 0.8);
                dVal = trapezoidalMF(dNorm, 0.0, 0.3, 0.7, 1.0);
                tVal = trapezoidalMF(tNorm, 0.0, 0.3, 0.7, 1.0);
                break;
            case 3: // Gaussian
                mVal = gaussianMF(mNorm, 0.5, 0.15);
                dVal = gaussianMF(dNorm, 0.5, 0.2);
                tVal = gaussianMF(tNorm, 0.5, 0.2);
                break;
            case 4: // Bell
                mVal = bellMF(mNorm, 0.1, 2, 0.5);
                dVal = bellMF(dNorm, 0.2, 2, 0.5);
                tVal = bellMF(tNorm, 0.2, 2, 0.5);
                break;
            case 5: // Sigmoid
                mVal = sigmoidMF(mNorm, 10, 0.5);
                dVal = sigmoidMF(dNorm, 10, 0.5);
                tVal = sigmoidMF(tNorm, 10, 0.5);
                break;
            case 6: // Left-Right
                mVal = leftRightMF(mNorm, 0.2, 0.4, 0.3);
                dVal = leftRightMF(dNorm, 0.2, 0.5, 0.2);
                tVal = leftRightMF(tNorm, 0.2, 0.5, 0.2);
                break;
            case 7: // Pi-shaped
                mVal = piMF(mNorm, 0.2, 0.4, 0.6, 0.8);
                dVal = piMF(dNorm, 0.0, 0.3, 0.7, 1.0);
                tVal = piMF(tNorm, 0.0, 0.3, 0.7, 1.0);
                break;
            case 8: // Open Left
                mVal = openLeftMF(mNorm, 0.2, 0.6);
                dVal = openLeftMF(dNorm, 0.3, 0.7);
                tVal = openLeftMF(tNorm, 0.3, 0.7);
                break;
            case 9: // Open Right
                mVal = openRightMF(mNorm, 0.2, 0.6);
                dVal = openRightMF(dNorm, 0.3, 0.7);
                tVal = openRightMF(tNorm, 0.3, 0.7);
                break;
            case 10: // S-Shaped
                mVal = sShapedMF(mNorm, 0.2, 0.8);
                dVal = sShapedMF(dNorm, 0.0, 1.0);
                tVal = sShapedMF(tNorm, 0.0, 1.0);
                break;
            default:
                System.out.println("Invalid choice!");
                return 0.0;
        }

        // Print individual MF values
        System.out.printf("Moisture MF Value = %.3f\n", mVal);
        System.out.printf("Days Stored MF Value = %.3f\n", dVal);
        System.out.printf("Temperature MF Value = %.3f\n", tVal);

        // Fuzzy rule: spoilage = max(moisture, days, temp)
        return Math.max(mVal, Math.max(dVal, tVal));
    }

    // ---------- MAIN ----------
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Input
        System.out.print("Enter Moisture % (0-100): ");
        double moisture = sc.nextDouble();
        System.out.print("Enter Days Stored (0-10): ");
        double days = sc.nextDouble();
        System.out.print("Enter Temperature in °C (0-20): ");
        double temp = sc.nextDouble();

        // Choose MF
        System.out.println("\nChoose Membership Function:");
        System.out.println(
                "1. Triangular\n2. Trapezoidal\n3. Gaussian\n4. Bell\n5. Sigmoid\n6. Left-Right\n7. Pi-shaped\n8. Open Left\n9. Open Right\n10. S-Shaped");
        System.out.print("Choose - ");
        int choice = sc.nextInt();

        // Compute spoilage
        double spoilage = fuzzySpoilage(moisture, days, temp, choice);

        // Output
        System.out.printf("\nSpoilage Level (0-1): %.3f\n", spoilage);
        if (spoilage < 0.3)
            System.out.println("Food is Fresh.");
        else if (spoilage < 0.7)
            System.out.println("Food is Moderately Spoiled.");
        else
            System.out.println("Food is Highly Spoiled!");
    }
}
