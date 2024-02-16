NLP
-RAs create more training data

-possible fix for relationships: represent as separate list on same level as people/events

-add a check to determine if entry is complete or not (to handle automatically transcribed partials)

-improve disambiguation logic (automate)
    -do manually for now, once you have a critical mass of data implement ML solution

-create records for all extant volumes, and upload to the cloud

HTR
-figure out the characteristics of images that ChatGPT can transcribe reliably

-improve (rewrite?) segmentation algo to produce this kind of data

-combine training data from Drive and S3 into a single bucket    
    -run upload script on directory with downloaded Drive content
    -write script to scan training data bucket, check for lines that have text but aren't in json master, and add (id/color scheme/text)

-add instructions and training data to transcription algo
    -make unmatched image segment removal part of HTR training data load

Production
-hook up the entire pipeline

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