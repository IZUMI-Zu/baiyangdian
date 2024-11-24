import logging
from predict import logger

def main():
    logger.info("Starting application...")
    try:
        # Your application code here
        pass
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise
    finally:
        logger.info("Application shutdown")

if __name__ == "__main__":
    # You can set additional logging configuration here if needed
    # For example, changing log level for specific modules:
    logging.getLogger("pandas").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    main()
