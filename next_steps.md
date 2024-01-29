-test how much training data is required and how granular it needs to be (metrics? do RAs need to do some manual extraction?)
    -create the necessary data for all extant transcriptions (RAs if we need a lot, this may also come before above)
    -possible fix for relationships: represent as separate list on same level as people/events

-improve disambiguation logic (automate)
    -do manually for now, once you have a critical mass of data implement ML solution

-create records for all extant volumes, and upload to the cloud

HTR

-play around with the GUI some to figure out what level of detail is best for HTR
    -lines? entries? entire images?

-create training data corresponding to this level of detail

-expand infrastructure to allow gpt4 api calls for text recognition
    -check pricing (or api documentation) for model name

-hook up the entire pipeline and see how widely applicable it is

-build lambda functions that trigger when IIIF manifests (?) hit an S3 bucket
    -need all images to already be in place and trigger needs to reference them

Production

-built api to access all images and extracted content

-built web interfaces to search and access this data

-write documentation for all of the above

Other possibilities:
-assign unique identifiers for places
    -pros: slightly easier to link data
    -cons: adds mostly unnecessary complexity to model instructions
-load comprehensive controlled vocabularies for various instructions
    -pros: if possible would improve model performance
    -cons: would add *a ton* of complexity, might literally be too many characters
    -this is something to consider more seriously once some use of the system has expanded the vocabularies
-build quick/easy NLP training data creation workflow
    -pros: would make training data creation easier
    -cons: may not need that much training data, would likely need to be a bespoke website