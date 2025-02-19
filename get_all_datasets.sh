#!bin/bash

# Run the dataset processing scripts, 1 by 1
#uv run process_dataset print_config=true  debug=false dataset=affectivetext
#uv run process_dataset print_config=false debug=false dataset=carer
#uv run process_dataset print_config=false debug=false dataset=crowdflower
#uv run process_dataset print_config=false debug=false dataset=emobank
#uv run process_dataset print_config=false debug=false dataset=emoint
#uv run process_dataset print_config=false debug=false dataset=fbvalencearousal
#uv run process_dataset print_config=false debug=false dataset=goemotions
#uv run process_dataset print_config=false debug=false dataset=sentimentalliar
#uv run process_dataset print_config=false debug=false dataset=ssec
#uv run process_dataset print_config=false debug=false dataset=talesemotions

uv run process_dataset print_config=false debug=false dataset=affectivetext
uv run process_dataset print_config=false debug=false dataset=carer
uv run process_dataset print_config=false debug=false dataset=crowdflower
uv run process_dataset print_config=false debug=false dataset=electoraltweets
uv run process_dataset print_config=false debug=false dataset=emobank
uv run process_dataset print_config=false debug=false dataset=emoint
uv run process_dataset print_config=false debug=false dataset=fbvalencearousal
uv run process_dataset print_config=false debug=false dataset=goemotions
uv run process_dataset print_config=false debug=false dataset=isear
uv run process_dataset print_config=false debug=false dataset=sentimentalliar
uv run process_dataset print_config=false debug=false dataset=ssec
uv run process_dataset print_config=false debug=false dataset=talesemotions
uv run process_dataset print_config=false debug=false dataset=xed

# This requires additional user input
#uv run process_dataset print_config=false debug=false dataset=ren20k

# Print the manifest of all the files created
uv run print_manifest
