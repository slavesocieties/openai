schema = {
    "type": "json_schema",
    "json_schema":
    {
        "name": "ssda_nlp",
        "schema": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "A unique identifier assigned to the person. These are alphanumeric identifiers of the P##, where ## is a two-digit padded integer and P01 is assigned to the first person to appear in the record, P02 to the second, etc."
                            },
                            "name": {
                                "type": "string",
                                "description": "The person's name exactly as it appears in the normalized text." 
                            },
                            "titles": {
                                "type": ["array"],
                                "items": {
                                    "type": "string",
                                    "description": "A list of any honorific titles accorded to the person exactly as they appear in the record."
                                }
                            },
                            "rank": {
                                "type": ["string", "null"],
                                "description": "The military rank earned by the person. These should be translated to their English equivalent."
                            },
                            "origin": {
                                "type": ["string", "null"],
                                "description": "The person's place of origin. This could be either an explicit place of birth or a vaguer declaration of origination. Continents, regions, or cities/towns are all acceptable."
                            },
                            "ethnicity": {
                                "type": ["string", "null"],
                                "description": "The ethnolinguistic descriptor assigned to the person in the record."
                            },
                            "age": {
                                "type": ["string", "null"],
                                "description": "A broad age category that applies to the person. These can either be explicit or implied based on referenced birth dates and should be recorded as their English equivalent. Note: The Spanish parvulo/a refers to an infant.",
                                "enum": ["infant", "child", "adult"]
                            },
                            "legitimate": {
                                "type": ["boolean", "null"],
                                "description": "The legitimacy of the person's birth. This will frequently be addressed explicitly, particularly in baptisms, but can also be implied (i.e. padres no conocidos). 'true' if legitimate, 'false' if not.",
                                "enum": [True, False]
                            },
                            "occupation": {
                                "type": ["string", "null"],
                                "description": "What the person did for a living. Most frequently this information will be available for the person who wrote the record. Occupations should be generalized (i.e. all clergymen are 'cleric's, regardless of position in the Church) and translated to English."
                            },
                            "phenotype": {
                                "type": ["string", "null"],
                                "description": "A physical/phenotypical descriptor assigned to the person in the record. Recorded as the male spelling of the variant regardless of person's sex.",
                                "enum": ["negro", "moreno", "indio", "pardo", "mestizo", "blanco", "criollo"]
                            },
                            "free": {
                                "type": ["boolean", "null"],
                                "description": "Whether or not the person was free at the time of the record's publication. This is a boolean field where true indicates freedom and false indicates enslavement.",
                                "enum": [True, False]
                            },
                            "relationships": {
                                "type": ["array"],
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "related_person": {
                                            "type": "string",
                                            "definition": "The unique identifier assigned to the other person in the relationship."
                                        },
                                        "relationship_type": {
                                            "type": "string",
                                            "definition": "What the other person is *to* this person. For example, if the normalized text describes the person with id P03 as 'hija de' the person with id P05, P05 would be recorded as 'parent' in P03's record and P03 would be recorded as 'child' in P05's record.",
                                            "enum": ["parent", "child", "enslaver", "slave", "spouse", "godparent", "godchild"]
                                        }
                                    },
                                    "required": ["related_person", "relationship_type"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["id", "name", "titles", "rank", "origin", "ethnicity", "age", "legitimate", "occupation", "phenotype", "free", "relationships"],
                        "additionalProperties": False
                    }                        
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "The type of event being recorded.",
                                "enum": ["baptism", "marriage", "burial", "birth"]
                            },
                            "principal": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "The unique identifier(s) assigned to the person(s) who the sacrament is being performed on/for. For baptisms, burials, and births this will be a single identifier, but for marriages it will be a two identifiers." 
                                }
                            },
                            "date": {
                                "type": ["string", "null"],
                                "description": "The date when the event took place. Represent dates in a YYYY-MM-DD format. If a complete date is not available, include as much information as possible."
                            }
                        },
                        "required": ["type", "principal", "date"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["people", "events"],
            "additionalProperties": False
        },
        "strict": True
    }
}