NLP
-RAs create more training data

-possible fix for relationships: represent as separate list on same level as people/events

-add a check to determine if entry is complete or not (to handle automatically transcribed partials)

-improve disambiguation logic (automate)
    -do manually for now, once you have a critical mass of data implement ML solution

-create records for all extant volumes, and upload to the cloud

HTR
-explore Assistant API?

-try upscaling images

-figure out the characteristics of images that ChatGPT can transcribe reliably

-improve (rewrite?) segmentation algo to produce this kind of data
    -make transcription GUI note coords, color scheme, and resolution and update training data log    

Production
-hook up the entire pipeline
    -scan training bucket before starting to build "official" data

-build lambda functions (?) that trigger when IIIF manifests (?) hit an S3 bucket
    -need all images to already be in place and trigger needs to reference them

-built api to access all images and extracted content

-built web interfaces to search and access this data

-write documentation for all of the above

Other possibilities:
-incorporate IIIF manifests into workflow
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