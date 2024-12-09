Agro Infection Detection Flask App



This project is a web application built using Flask and TensorFlow. It allows users to upload images of plant leaves and predicts the disease based on a pre-trained model.




Features:
Upload an image of a plant leaf.
The model will predict the disease in the leaf and display the result on the webpage.
The model was trained using a custom dataset of plant diseases.





Requirements:
Python 3.x
Flask
TensorFlow
OpenCV
werkzeug
numpy




Setup Instructions:

Create a new folder
cd <folder_name>


Step 2:Clone the repository



Clone this repository to your local machine using:


git clone https://github.com/riyap2003/agroinfectiondetection.git



Step 2: Install Dependencies




Ensure that you have Python 3.x installed. Then, install the required libraries using pip:



pip install -r requirements.txt



OR



If requirements.txt is not provided, you can manually install the dependencies:



pip install flask tensorflow opencv-python werkzeug numpy


Step 3: Download the Dataset



You can download the dataset from Kaggle's Plant Disease Dataset. Alternatively, you can use your own dataset of plant leaf diseases. If you're using a custom dataset, make sure to update the model file path and class names accordingly.



Step 4: Model
The Flask app expects the trained model to be saved as model/trained_model.keras.
You can train your model using the dataset or use a pre-trained model that is in my repo



Step 5: Run the Application
Once all the dependencies are installed, and the model is set up, you can run the Flask application using:



python app.py


The app will start a local server. You can open the browser and go to http://127.0.0.1:5000 to access the application.



Step 6: Upload an Image



On the home page, you can upload an image of a plant leaf.
After uploading the image, the app will process it, use the trained model to predict the plant disease, and display the result.







agro/

|

├── .gitignore               # Specifies which files/folders to exclude from Git tracking

|

├── README.md                # Project description, installation instructions, etc.

|

├── app.py                    # Flask application for plant disease prediction

|

├── model/                    # Folder to store your trained model

|   |

│   └── trained_model.keras   # Your trained Keras model

|

├── static/                   # Static files (images, CSS, JS)

|   |

│   ├── image.png             # Static image

|

│   ├── style.css             # Stylesheet

|   |

│   └── uploads/              # Folder for uploaded images

|       |

│       ├── AppleCedarRust1.JPG

|       |

│       ├── AppleCedarRust2.JPG

|       |

│       └── ...               # Other uploaded images

|

├── templates/                # HTML templates for the Flask app

|

│   └── index.html            # Main HTML page for the web app

|

├── test/                     # Folder containing test images for validation

|   |

│   ├── AppleCedarRust1.JPG

|   |

│   ├── CornCommonRust1.JPG

|   |

│   └── ...                   # Other test images

|

├── train/                    # Folder for training images

|   |

│   └── ...                   # Training images

|

├── valid/                    # Folder for validation images

|   |

│   └── ...                   # Validation images

|

└── flask_env/                # Virtual environment folder (should be in .gitignore)








