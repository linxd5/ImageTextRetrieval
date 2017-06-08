/**
 * Created by lindayong on 17-5-9.
 */

$(function () {
    $('#query_button').click(query);
    $('#reset_button').click(function () {
        $('form')[0].reset();
        $('#upload_img').remove();
    });

    function query() {
        // var formData = new FormData(document.querySelector('form'));
        var formData = new FormData($('form')[0]);
        var query_button = $('#query_button');
        query_button.attr('disabled', 'true');
        query_button.text('正在查询中，请稍后 ...');
        $.ajax({
            url: '/query',
            data: formData,
            type: 'POST',
            processData: false,
            contentType: false,
            success: function (response) {
                $('#img_result').empty();
                $('#upload_img').remove();
                $('#text_result').empty();
                query_button.removeAttr('disabled');
                query_button.text('查询');
                console.log(response);
                var img_dirs = response['sim_images'];
                var upload_img = response['upload_img'];
                var texts = response['sim_texts'];
                if (upload_img !== 'no_upload_img') {
                    upload_img = '<img src="' + upload_img + '" id="upload_img" alt="upload_img"/>';
                    $('#query_image').after(upload_img)
                }
                for (num in img_dirs) {
                    var img_dir = '../'+img_dirs[num].replace(/%/g, '%25');
                    var img_html = '<img src="' + img_dir + '" />';
                    $('#img_result').append(img_html);
                }
                for (num in texts) {
                    var text_html = '<div>' + texts[num] + '</div>';
                    $('#text_result').append(text_html);
                }
            },
            error: function (error) {
                console.log(error);
            }
        });
    }
});
