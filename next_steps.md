-test how much training data is required and how granular it needs to be (metrics? do RAs need to do some manual extraction?)
    -create the necessary data for all extant transcriptions (RAs if we need a lot, this may also come before above)
    -possible fix for relationships: represent as separate list on same level as people/events

-add a check to determine if entry is complete or not (to handle automatically transcribed partials)

-improve disambiguation logic (automate)
    -do manually for now, once you have a critical mass of data implement ML solution

-create records for all extant volumes, and upload to the cloud

HTR

-update segmentation app to return entry counts and lines per entry

(-combine training data from Drive and S3 into a single bucket, make sure that color
 scheme of training data is consistent and corresponds to input)

-expand infrastructure to allow gpt4 api calls for text recognition
    -can we detect and handle bad segmentation adequately?    

-hook up the entire pipeline and see how widely applicable it is

-build lambda functions (?) that trigger when IIIF manifests (?) hit an S3 bucket
    -need all images to already be in place and trigger needs to reference them

Production

-built api to access all images and extracted content

-built web interfaces to search and access this data

-write documentation for all of the above

Other possibilities:
-incorporate IIIF manifests into workflow
-play around with the GUI some to figure out what level of detail is best for HTR    
    -in the short term I'm going to build this with lines since the model won't start
     making stuff up and we have good training data
    -however, in the long term this will not be the most cost-effective solution,
     so should eventually investigate if we can do this on the level of entries
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