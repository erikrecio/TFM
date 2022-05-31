
function test(change)
    #expr = :(change * " = " * change)
    expr = :(change)
    @eval $expr
end

function test2(i)
    @eval $i
  end

function main()
    global iter = 10
    change = "iter"

    str_change = "\"" * change * " = \$" * change * "" * "\""
    str_change_parse = Meta.parse(str_change)
    str_change_eval = eval(str_change_parse)

    print(str_change)
    print("\n")
    print(str_change_parse)
    print("\n")
    print(str_change_eval)
    print("\n")
    print("iter = $(iter)")
    print("\n")

    #expr = :(change * " = " * change)
    #a = test(change)
    #print(a)
    i = change
    print(i)
    test2(i)

    exprtoeval=:(x*x)
    @eval f(x)=$exprtoeval
    f(4) # => 16
    print("\n")

    exprtoeval=:("\"" * x * " = " * x * "" * "\"")
    @eval f(x) = $exprtoeval
    f(change)

end  

function main2()

    global a = 5
    change = "a"

    str_change = "\"" * change * " = \$" * change * "" * "\""
    str_change_parse = Meta.parse(str_change)

    str_change_eval = eval(str_change_parse)
    print(str_change_eval)

    for a in range(10,10)
        str_change_eval = eval(str_change_parse)
        print(str_change_eval)
    end
end

function main3()

    global alehop = 1
    a = :epaakhgktc
    print(Symbol(a))
    print("\n")
    print(a)

end


main3()