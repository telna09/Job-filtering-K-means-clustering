import logging
from job_system import JobClassificationSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_menu():
    print("\n--- Job Classification System ---")
    print("1. Train Model")
    print("2. Match Jobs to User Preferences")
    print("3. Send Email Alert")
    print("4. Exit")

def main():
    system = JobClassificationSystem()
    model_data = None
    matched_jobs = []

    while True:
        display_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            terms_input = input("Enter search terms (comma-separated, default: Software Engineer): ").strip()
            search_terms = [term.strip() for term in terms_input.split(',')] if terms_input else ['Software Engineer']
            model_data = system.train_model(search_terms=search_terms)
            if model_data:
                logger.info("Model trained successfully.")
            else:
                logger.warning("Model training failed or returned no data.")

        elif choice == '2':
            if not model_data:
                logger.warning("Please train the model first.")
                continue

            preferences = input("Enter your preferences (comma-separated skills): ").strip()
            user_skills = [skill.strip() for skill in preferences.split(',')]
            matched_jobs = system.match_user(user_skills)
            if matched_jobs:
                print("\nMatched Jobs:")
                for job in matched_jobs:
                    print(f"{job['title']} at {job['company']} ({job['location']})")
                    print(f"  Skills: {job['skills']}")
                    print(f"  URL: {job['url']}\n")
            else:
                print("No jobs matched your preferences.")

        elif choice == '3':
            if not matched_jobs:
                logger.warning("No matched jobs to send. Match jobs first.")
                continue

            to_email = input("Enter recipient email address: ").strip()
            system.send_alerts(matched_jobs, to_email)
            print("Email alert sent successfully.")

        elif choice == '4':
            print("Exiting the application.")
            break

        else:
            print("Invalid choice. Please select from 1 to 4.")

if __name__ == "__main__":
    main()
