Once upon a time, in life before NTG, I ran a small consulting business.  As part of this business, I was contracted by a non-profit to build a "prayer request" website.  I tapped into the database holding the requests this past weekend and pulled everything that was available, roughly 22K records.

I exported these records into a Json file.  To keep it simple, I only exported the text of each record and no supporting data, like times or email addresses, or names, or etc...

So, the JSON Looks like:

[
{"Request":"Request 1"},
{"Request":"Request 2"},
...
]

The data file is included as dd.json


Relying very much on ChatGPT, I asked it to create a SLM from the above described JSON structure.

That conversation is captured here:  https://chatgpt.com/share/1cacf763-dd90-4f47-8714-0b2e6360b659

The files included in this repository are a Visual Studio Solution using Python to execute the steps described in the conversation I had with ChatGPT

The important python files are:

LMProcessingProcessor.py - the script used to generate the LM
runmodel.py - the script written to interact with the model

Add two directories:
	fine_tuned_model
	results

These are outputs for the entire process but the files, including the model, were too large for Git.  You can download them from http://www.christophermeadows.com/output.zip so you don't have to rebuild the model

The training took 9 hours, 21 minutes and 28 seconds on my personal computer (x64 Windows 11 Home 32GM RAM, 11th gen i7 @2.90GHz, 4 core, Solid State HD)

Check out ModelTraining.png for training results (the view was captured after the process finished - while running the CPU stayed at a constant 95% utilization)

As best as I can understand, my model was trained on very basic, old, model standards so the results would probably be nowhere close to a production level service but it's a start (at least to my learning).