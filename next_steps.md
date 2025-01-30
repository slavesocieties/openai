-strip down segmentation algo to its component parts and rebuild in a cleaner way
    -and/or pursue a different segmentation solution
-improve segmentation algo
    -assess performance across wider range of volumes    
    -add a separate pre-processing step to crop to folio?
    -add hyper-parameters to driver
    -add verbosity to other pipeline components
    -improve ability to detect horizontal breaks between sections and prevent false positives

NLP
    -why are some people ids padded and others not?
    -possible fix for relationships: represent as separate list on same level as people/events
    -improve check to determine if entry is complete or not
        -will likely need training data, but can be simple
        -should likely also spend some time improving prompts
    -improve disambiguation logic (automate)
        -do manually for now, once you have a critical mass of data implement ML solution
    -add Corrina's training data to dataset
    -build more training data
    -assess performance for different kinds of records

HTR
    -build more training data
    -determine how granular model tuning needs to be
    -incorporate name list checks for people, places, characteristics, etc.
    -teach model to indicate uncertainty (improve prompts)    

Production
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