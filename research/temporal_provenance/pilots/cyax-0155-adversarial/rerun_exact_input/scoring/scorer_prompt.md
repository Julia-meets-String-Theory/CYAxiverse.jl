# CYAX-0163 blind scoring instructions

Score the six opaque responses below against the frozen answer key and exact
scorecard schema. Use only the supplied answer key, schema, and response text.
Do not use tools, files, web access, prior conversation state, or any outside
source. Do not infer which opaque response came from which experimental
condition. Preserve uncertainty and score only claims supported by the
response and the frozen key.

Return exactly six JSON objects, one per line, in opaque-id order S1 through
S6. Do not use Markdown fences, commentary, extra fields, or a surrounding
array. Every object must validate against the supplied scorecard schema. Set
`source_reopenings` to zero unless the response itself explicitly reports a
source reopening, and record automatic-failure reasons separately from the
raw rubric score.
