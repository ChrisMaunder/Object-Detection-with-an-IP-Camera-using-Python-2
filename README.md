# Object Detection with an IP Camera using Python and CodeProject.AI Server, Part 2

The second in a two-part series on detecting objects and evil rodents

![Scheming Racoon](https://raw.githubusercontent.com/ChrisMaunder/Object-Detection-with-an-IP-Camera-using-Python-2/master/docs/assets/scheming_raccoon_detected.jpg)

## Introduction

In our previous article, [Detecting raccoons using CodeProject.AI Server, Part 1](/Articles/5344693/Object-Detection-with-an-IP-camera-using-Python-an), we showed how to hook-up the video stream from a Wyze camera and send that to CodeProject.AI Server in order to detect objects.

In this article, we will train our own model specifically for raccoons and setup a simple alert that will tell us when one of these trash pandas is on a mission.

## Training a Model for CodeProject.AI Server Object Detection

CodeProject.AI Server comes, out of the box, with multiple Object Detection modules available. We'll focus on the YOLOv5 6.2 module to keep things simple, which means training a YOLOv5 PyTorch model.

### Setup

This article is based on Matthew Dennis' the far more comprehensive article, [How to Train a Custom YOLOv5 Model to Detect Objects](/Articles/5347827/How-to-Train-a-Custom-YOLOv5-Model-to-Detect-Objec). We'll summarise the setup quickly just to get you up to speed.

Start with Visual Studio Code, with the [Jupyter notebook extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), and the [Jupyter notebook](/Articles/5347827/How-to-Train-a-Custom-YOLOv5-Model-to-Detect-Objec) that Matthew's article supplies. In that notebook, a Python Virtual environment is created.

```cpp
!Python -m venv venv
```

Clone the Ultralytics YOLOv5 repository

```cpp
!git clone <a href="https://github.com/ultralytics/yolov5">https://github.com/ultralytics/yolov5</a>
```

and set the Virtual Environment that VS Code will use by choosing 'venv' in the list of environments in the upper right the notebook.

In the download containing the Jupyter notebook is a *requirements.txt* file that contains the dependencies you will need to install when setting up your Python environment.

Install the dependencies:

```cpp
%pip install fiftyone
%pip install -r requirements-cpu.txt 
%pip install ipywidgets
```

Use *requirements-gpu.txt* if you have an NVIDIA GPU.

### The Training Data

To train a custom model, we need images with which to build the model. An important point in selecting training data and building a model like ours, where we only wish to detect raccoons, is that we need to be sure we train for raccoons, and not squirrels, cats, or very, very small bears. To do this, we will train a model on not just raccoons but dogs, cats, squirrels, skunks and raccoons.

We'll use the excellent fiftyone package to grab images from the extensive Open Images collection. To start with, we create a critters dataset containing just raccoons, then iteratively add cats, dogs, and the rest to this collection.

```cpp
    import fiftyone as fo
    import fiftyone.zoo as foz

    splits = ["train", "validation", "test"]
    numSamples = 10000
    seed = 42

    # Get 10,000 images (maybe in total, maybe of each split) from fiftyone. 
    # We'll ask FiftyOne to use images from the open-images-v6 dataset and 
    # store information of this download in the dataset named 
    # "open-imges-critters". 

    # The data that's downloaded will include the images, annotations, and
    # a summary of what's been downloaded. That summary will be stored 
    # /Users/<username>/.FiftyOne in a mongoDB database. The images / 
    # annotations will be in /Users/<username>/FiftyOne.

    if fo.dataset_exists("open-images-critters"):
        fo.delete_dataset("open-images-critters")

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        splits=splits,
        label_types=["detections"],
        classes="Raccoon",
        max_samples=numSamples,
        seed=seed,
        shuffle=True,
        dataset_name="open-images-critters")

    # Take a quick peek to see what's there
    print(dataset)

    # Do the same for cats, dogs, squirrels, and skunks, but after each
    # download we'll merge the new downloaded dataset with the existing 
    # open-images-critters dataset so we can build up one large, 
    # multi-class set

    if fo.dataset_exists("open-images-cats"):
        fo.delete_dataset("open-images-cats")

    cats_dataset = foz.load_zoo_dataset(
        "open-images-v6",
        splits=splits,
        label_types=["detections"],
        classes="Cat",
        max_samples=numSamples,
        seed=seed,
        shuffle=True,
        dataset_name="open-images-cats")

    # Now merge this new set with the existing open-images-critters set
    dataset.merge_samples(cats_dataset)

    if fo.dataset_exists("open-images-dogs"):
        fo.delete_dataset("open-images-dogs")

    dogs_dataset = foz.load_zoo_dataset(
        "open-images-v6",
        splits=splits,
        label_types=["detections"],
        classes="Dog",
        max_samples=numSamples,
        seed=seed,
        shuffle=True,
        dataset_name="open-images-dogs")

    dataset.merge_samples(dogs_dataset)

    if fo.dataset_exists("open-images-squirrels"):
        fo.delete_dataset("open-images-squirrels")

    squirrels_dataset = foz.load_zoo_dataset(
        "open-images-v6",
        splits=splits,
        label_types=["detections"],
        classes="Squirrel",
        max_samples=numSamples,
        seed=seed,
        shuffle=True,
        dataset_name="open-images-squirrels")

    dataset.merge_samples(squirrels_dataset)

    if fo.dataset_exists("open-images-skunks"):
        fo.delete_dataset("open-images-skunks")

    skunks_dataset = foz.load_zoo_dataset(
        "open-images-v6",
        splits=splits,
        label_types=["detections"],
        classes="Skunk",
        max_samples=numSamples,
        seed=seed,
        shuffle=True,
        dataset_name="open-images-skunks")

    dataset.merge_samples(skunks_dataset)

    # For whenever you want to see what's been loaded.
    print(fo.list_datasets())

    # Uncomment the following line if you wish to explore the 
    # resulting datasets in the FiftyOne UI
    # session = fo.launch_app(dataset, port=5151)
```

The next step is to export this training data to the format required by the YOLOv5 trainers:

```cpp
    import fiftyone as fo

    export_dir = "datasets/critters"
    label_field = "detections"  # for example

    # The splits to export
    splits = ["train", "validation","test"]

    # All splits must use the same classes list
    classes = ["Raccoon", "Cat", "Dog", "Squirrel", "Skunk"]

    # The dataset or view to export
    # We assume the dataset uses sample tags to encode the splits to export
    dataset_or_view = fo.load_dataset("open-images-critters")

    # Export the splits
    for split in splits:
        split_view = dataset_or_view.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=classes,
        )
```

During this process, a *datasets\critters\dataset.yaml* file will be created. We'll need to tweak this slightly to rename `validation` to `val`. Your file should look like:

```cpp
names: 
- Raccoon 
- Cat 
- Dog 
- Squirrel 
- Skunk 
nc: 5 
path: c:\Dev\YoloV5_Training\datasets\critters 
train: .\images\train\ 
test: .\images\test\ 
val: .\images\validation\
```

`nc` is "Number of classes", which is `5` (Racoon, Cat, Dog, Squirrel, Skunk), `path` is the path to the images, and `train`, `test`, and `val` are the folders containing the training, test, and validation data for our mode training process.

#### A note on the number of images

The two ways to improve a model's accuracy are:

1. Train for longer (more 'epochs', or iterations of the training)
2. Train with more data (more images)

You may want to adjust the number of images to suit your setup. Resource use can be considerable, and the more images, the longer to train. With 50 epochs and 1,000 images, training on an NVIDIA 3060 GPU takes about 50 minutes. 25,000 images and 300 epochs takes around 30 hours.

## Training the Model

To start the model training within our Jupyter notebook, we run *yolov5/train.py* Python module using the ! syntax to launch an external process:

```cpp
!python yolov5/train.py --batch 24 --weights
yolov5s.pt --data datasets/critters/dataset.yaml --project train/critters
--name epochs50 --epochs 300
```

We set the `batch` parameter to `24` simply to ensure we didn't run out of memory. We have 16GB system memory, 12GB dedicated GPU memory. With a smaller (1,000 image) dataset, a batch of 32 was OK, but with a larger image set, a batch of 32 was too high. You may have to experiment to get the optimal batch size for your machine.

### Interrupting and Restarting the Training

You can stop the training at any point and restart using the `--resume` flag

```cpp
!python yolov5/train.py --resume train/critters/epochs300/weights/last.pt 
```

## Using our Model

Grab the *critters.pt* file that was created by our training and drop it in *C:\Program Files\CodeProject\AI\modules\ObjectDetectionYolo\custom-models*. CodeProject.AI server will immediately be able to use this new model without any changes or restarts using the route *vision/custom/critters*, **as long as you are using the YOLO 6.2 module**. Each module has its own custom model location.

We can test by opening up the [CodeProject.AI Server Explorer](http://localhost:32168/vision.html) that's installed as part of CodeProject.AI. Choose the **Vision** tab, select an image of a raccoon next to the **Custom Detect** button, choose 'critters' as the **Model**, and test.

![Gotcha](https://raw.githubusercontent.com/ChrisMaunder/Object-Detection-with-an-IP-Camera-using-Python-2/master/docs/assets/test_raccoon_detect.jpg)

## Updating our Wyze Cam Code to Use this New Model

We'll be modifying the code from [Detecting raccoons using CodeProject.AI Server Part 1](/Articles/5344693/Object-Detection-with-an-IP-camera-using-Python-an) to add two things:

1. We'll use our new model
2. We'll fire an alert when a racoon is detected

### Using the Model

Using the model is trivial. We will modify our `do_detection` method to use the new model by changing the line in `do_detection` from:

```cpp
        response = session.post(opts.endpoint("vision/detection"), 
```

to:

```cpp
        response = session.post(opts.endpoint("vision/custom/critters"), 
```

However, to wire up an alert, we need to know what to look for, and whether it was found. We'll add a parameter that accepts a list of 'intruders' to watch for, and also return a comma delimited list of intruders found.

```cpp
model_name = "critters"             # Model we'll use
intruders  = [ "racoon", "skunk" ]  # Things we care about

def do_detection(image: Image, intruders: List[str]) -> "(Image, str)":

    """
    Performs object detection on an image and returns an image with the objects
    that were detected outlined, as well as a de-duped list of objects detected.
    If nothing detected, image and list of objects are both returned as None
    """

    # Convert to format suitable for a POST
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    buf.seek(0)

    # Better to have a session object created once at the start and closed at
    # the end, but we keep the code simpler here for demo purposes    
    with requests.Session() as session:
        response = session.post(opts.endpoint("vision/custom/" + model_name),
                                files={"image": ('image.png', buf, 'image/png') },
                                data={"min_confidence": 0.5}).json()

    # Get the predictions (but be careful of a null return)
    predictions = response["predictions"]

    detected_list = []

    if predictions:
        # Draw each bounding box that was returned by the AI engine
        # font = ImageFont.load_default()
        font_size = 25
        padding   = 5
        font = ImageFont.truetype("arial.ttf", font_size)
        draw = ImageDraw.Draw(image)

        for object in predictions:
            label = object["label"]
            conf  = object["confidence"]
            y_max = int(object["y_max"])
            y_min = int(object["y_min"])
            x_max = int(object["x_max"])
            x_min = int(object["x_min"])

            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=5)
            draw.rectangle([(x_min, y_min - 2*padding - font_size), 
                            (x_max, y_min)], fill="red", outline="red")
            draw.text((x_min + padding, y_min - padding - font_size),
                       f"{label} {round(conf*100.0,0)}%", font=font)

            # We're looking for specific objects. Build a deduped list
            # containing only the objects we're interested in.
            if label in intruders and not label in detected_list:
                detected_list.append(label)

    # All done. Did we find any objects we were interested in?
    if detected_list:
        return image, ', '.join(detected_list)

    return None, None 
```

Next we'll modify the `main` method so that if we have detected a raccoon, an alert is thrown.

```cpp
secs_between_checks = 5   # Min secs between sending a frame to CodeProject.AI
last_check_time = datetime(1999, 11, 15, 0, 0, 0)
recipient       = "alerts@acme_security.com"    # Sucker who deals with reports

def main():

    # Open the RTSP stream
    vs = VideoStream(opts.rtsp_url).start() 

    while True:

        # Grab a frame at a time
        frame = vs.read()
        if frame is None:
            continue

        objects_detected = ""

        # Let's not send an alert *every* time we see an object, otherwise we'll
        # get an endless stream of emails, fractions of a second apart
        global last_check_time
        seconds_since_last_check = (datetime.now() - last_check_time).total_seconds()

        if seconds_since_last_check >= secs_between_checks:
            # You may need to convert the colour space.
            # image: Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image: Image = Image.fromarray(frame)
            (image, objects_detected) = do_detection(image, intruders)

            # Replace the webcam feed's frame with our image that include object 
            # bounding boxes
            if image:
                frame = np.asarray(image)

            last_check_time = datetime.now()

        # Resize and display the frame on the screen
        if frame is not None:
            frame = imutils.resize(frame, width = 1200)
            cv2.imshow('WyzeCam', frame)

            if objects_detected:
                # Shrink the image to reduce email size
                frame = imutils.resize(frame, width = 600)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                report_intruder(image, objects_detected, recipient)

        # Wait for the user to hit 'q' for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up and we're outta here.
    cv2.destroyAllWindows()
    vs.stop() 
```

Note that we're not sending every single frame to CodeProject.AI. That would chew up a fair bit of processor time and isn't necessary. A Wyze cam runs at 15 frames a second, but to be practical, we can probably do with checking a frame every few seconds. Season to taste.

The last piece of the puzzle is the `report_intruder` method. We'll write to console the list of intruders detected, as well as send an email to whomever needs to know. For email, we're using a Gmail account.

To enable this, use or create a Gmail account and use the Windows `setx` command to store the email and password for your account in an environment variable. **This is not secure** but it beats committing your password to a Git repo. Please use a test email account for this, not your actual email account.

```cpp
setx CPAI_EMAIL_DEMO_FROM "me@gmail.com"
setx CPAI_EMAIL_DEMO_PWD  "password123" 
```

Our `report_intruder` method, and the `send_email` method it uses, are as follows:

```cpp
last_alert_time = datetime(1999, 11, 15, 0, 0, 0)
secs_between_alerts = 300 # Min secs between sending alerts (don't spam!)

def report_intruder(image: Image, objects_detected: str, recipient: str) -> None:

    # time since we last sent an alert
    global last_alert_time
    seconds_since_last_alert = (datetime.now() - last_alert_time).total_seconds()

    # Only send an alert if there's been sufficient time since the last alert
    if seconds_since_last_alert > secs_between_alerts:

        # Simple console output
        timestamp = datetime.now().strftime("%d %b %Y %I:%M:%S %p")
        print(f"{timestamp} Intruder or intruders detected: {objects_detected}")

        # Send an email alert as well
        with BytesIO() as buffered:
            image.save(buffered, format="JPEG")
            img_dataB64_bytes : bytes = base64.b64encode(buffered.getvalue())
            img_dataB64 : str = img_dataB64_bytes.decode("ascii");

        message_html = "<p>An intruder was detected. Please review this image</p>" \
                     + f"<img src='data:image/jpeg;base64,{img_dataB64}'>"
        message_text = "A intruder was detected. We're all doomed!"

        send_email(opts.email_acct, opts.email_pwd, recipient, "Intruder Alert!", 
                   message_text, message_html)

        # Could send an SMS or a tweet. Whatever takes your fancy...

        last_alert_time = datetime.now()

def send_email(sender, pwd, recipient, subject, message_text, message_html):

    msg = MIMEMultipart('alternative')
    msg['From']    = sender
    msg['To']      = recipient
    msg['Subject'] = subject

    text = MIMEText(message_text, 'plain')
    html = MIMEText(message_html, 'html')
    msg.attach(text)
    msg.attach(html)

    try:
        server = smtplib.SMTP(opts.email_server, opts.email_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, pwd)
        server.send_message(msg, sender, [recipient])
    except Exception as ex:
        print(f"Error sending email: {ex}")
    finally:
        server.quit() 
```

## Conclusion

We've walked through taking a stock Wyze cam and updating its firmware we're able to access the RTSP stream for processing. We've then used then Open Images dataset to create a custom YOLOv5 model for detecting critters. By adding this model to the *custom-models* folder, of the YOLOv5 6.2 Object Detection module in CodeProjet.AI Server, we have our very own raccoon detector. A little more Python and we can use this detector to regularly check our Wyze cam's feed and send us an email when one of the little masked bandits comes into view.

The code is included in the CodeProject.AI Server source code (in *Demos/Python/ObjectDetect/racoon\_detect.py*).

We wrote CodeProject.AI Server to take away the pain of setting up AI systems and projects. We deal with the runtimes, packages and getting all the pieces in place so we can skip straight to the fun parts like detecting trash pandas.

Please [download CodeProject.AI](https://www.codeproject.com/Articles/5322557/CodeProject-AI-Server-AI-the-easy-way) and give it a go. Add your own modules, integrate it with your apps, train some custom models and use it to learn a little about Artificial Intelligence.
