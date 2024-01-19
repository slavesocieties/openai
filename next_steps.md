-test how much training data is required and how granular it needs to be (metrics? do RAs need to do some manual extraction?)
    -create the necessary data for all extant transcriptions (RAs if we need a lot, this may also come before above)

-fully build out infrastructure to extract data from a full volume, fill in missing dates, disambiguate, etc.
 and create master volume record

For later:
-build an endpoint that can be reached remotely, write basic documentation
    -lambda api won't work, timeout is too short
    -is this actually something that we need to expose an api for?

Other possibilities:
-assign unique identifiers for places
    -pros: slightly easier to link data
    -cons: adds mostly unnecessary complexity to model instructions
-load comprehensive controlled vocabularies for various instructions
    -pros: if possible would improve model performance
    -cons: would add *a ton* of complexity, might literally be too many characters