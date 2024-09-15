
# use builder pattern to construct the hierarchicalMH, which itself is
# similar to a composite pattern, consisting of Metropolis Hastings Chains
# corresponding to a hierarchy of Bayesian models. Ideally, I can wrap all
# of this up in a hierarchical Proposal method and plug this into the 
# usual Metropolised Random Walk class, which represents the chain on the
# finest level. Could be a good idea to start with the fine chain and work
# my way down from there through the proposal and builder.
 
# sidenote: I think what I am actually doing in the chainFactory is more akin
# to a builder pattern than to a factory pattern.
