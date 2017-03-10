
def get_stock_price(sample_array):
    output = []
    for i, v in enumerate(sample_array):
        count = 1
        for j in sample_array[0:i]:
            if j <= v:
                count = count + 1
            else:
                count = 1
        output.append(count)
    print output
    return output

# Linear time


def get_stock_price_linear(m):
    op = []
    p = []
    max_at = 0

    for i in range(0, len(m)):

        print 'output', op
        if i == 0:
            op.append(1)
            p.append([m[i]])
            x = m[i]
            continue

        if m[i - 1] <= m[i]:
            p[-1].append(m[i])
        else:
            p.append([m[i]])
        u = len(p[-1]) - 1

        if m[i] >= x:
            u = u + op[max_at]
            max_at = i

        x = max(x, m[i - 1], m[i])

        op.append(u)

        print 'P', p
        print 'max_at', max_at
        print 'max_till_now', x

    print 'output', op
    return op

if __name__ == '__main__':
    # sample_array = [100, 80, 60, 70, 60, 75, 85]
    sample_array = [34, 56, 35, 85, 10, 90, 100, 101, 30]
    print sample_array
    output_array = [1, 2, 1, 4, 1, 6]
    print 'output_array expected', output_array
    outpute = get_stock_price_linear(sample_array)
    # outpute = get_stock_price(sample_array)
    assert(outpute == output_array)
